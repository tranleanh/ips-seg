import os
import cv2
import logging
import argparse
import numpy as np

from nets import utils
from ultralytics.models.sam import SAM3SemanticPredictor


# ------------------------------------ #
#             CONFIGURATION
# ------------------------------------ #
logging.getLogger("ultralytics").disabled = True

# Parse arguments
parser = argparse.ArgumentParser(description="Argument parser for SAM3.")
parser.add_argument("--in_dir", type=str, default='imgs/inputs', help="Input folder")
parser.add_argument("--out_dir", type=str, default='imgs/inputs_pred', help="Output folder")
parser.add_argument("--model_path", type=str, default='models/sam3.pt', help="Model path")
parser.add_argument("--sam3_conf", type=float, default=0.25, help="SAM3 confidence threshold")
parser.add_argument("--save_crop", type=bool, default=False, help="Saving cropped patches")
parser.add_argument("--prompts", default=["drone", "uav", "flying drone", "quadcopter", "unmanned aerial vehicle"], help="Context prompts")
args = parser.parse_args()

IMAGE_FOLDER  = args.in_dir
OUTPUT_DIR = args.out_dir
TEXT_PROMPTS = args.prompts
SAM3_MODEL_PATH = args.model_path
SAM3_CONF = args.sam3_conf
SAVE_CROP = args.save_crop

MIN_BBOX_AREA = 50
PADDING = 20 
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------ #
#               EXECUTION
# ------------------------------------ #
print("Initializing SAM3...")
sam3_overrides = dict(
    conf=SAM3_CONF, 
    task="segment", 
    mode="predict", 
    model=SAM3_MODEL_PATH, 
    half=True,    # Set to False if running on CPU
    device="cuda", # Explicitly forcing GPU execution
    save=False,
    verbose=False
)
sam3_predictor = SAM3SemanticPredictor(overrides=sam3_overrides)
print("SAM3 initialized successfully!")

# Get all image files
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]

print("\nStarting inference...")
for idx, filename in enumerate(all_files, 1):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    filename_mask = filename.replace('jpg', 'png')
    fname = os.path.splitext(filename)[0]
    
    # Load Image
    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]

    # ---------------------------------------------------------------
    # STAGE 1: SAM3 Inference
    sam3_predictor.set_image(image_path)
    coarse_results = sam3_predictor(text=TEXT_PROMPTS)

    # Coarse Mask
    coarse_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for res in coarse_results:
        if res.masks is not None:
            for mask in res.masks.data:
                m_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                if m_np.shape != (h_img, w_img):
                    m_np = cv2.resize(m_np, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                coarse_mask = cv2.bitwise_or(coarse_mask, m_np)

    # Save coarse mask
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_sam3.png"), coarse_mask)

    # ---------------------------------------------------------------
    # STAGE 2: Crop and Refine
    bboxes = utils.get_bboxes(coarse_mask, min_area=MIN_BBOX_AREA)
    final_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    for b_idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        x1_p, y1_p = max(0, x1 - PADDING), max(0, y1 - PADDING)
        x2_p, y2_p = min(w_img, x2 + PADDING), min(h_img, y2 + PADDING)

        crop = image[y1_p:y2_p, x1_p:x2_p].copy()
        if crop.size == 0: continue
        
        # Save crop temporarily for predictor
        temp_path = f"temp_crop.png"
        cv2.imwrite(temp_path, crop)

        # Save crop
        if SAVE_CROP:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_p{b_idx}.jpg"), crop)

        sam3_predictor.set_image(temp_path)
        refined_results = sam3_predictor(text=TEXT_PROMPTS)
        
        h_c, w_c = crop.shape[:2]
        current_crop_mask = np.zeros((h_c, w_c), dtype=np.uint8)

        for res in refined_results:
            if res.masks is not None:
                for mask in res.masks.data:
                    m_c = (mask.cpu().numpy() * 255).astype(np.uint8)
                    if m_c.shape != (h_c, w_c):
                        m_c = cv2.resize(m_c, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
                    current_crop_mask = cv2.bitwise_or(current_crop_mask, m_c)

        # Save predited mask for the crop
        if SAVE_CROP:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_p{b_idx}_mask.png"), current_crop_mask)

        # Map back to final global mask
        final_mask[y1_p:y2_p, x1_p:x2_p] = cv2.bitwise_or(final_mask[y1_p:y2_p, x1_p:x2_p], current_crop_mask)

    pred_mask = final_mask.copy()
    # ---------------------------------------------------------------

    print(f"Processed {idx}/{len(all_files)}: {filename}")

    # Save fine mask
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_sam3_2s.png"), pred_mask)

print(f'Outputs saved to {OUTPUT_DIR}')