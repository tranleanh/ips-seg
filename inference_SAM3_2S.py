import os
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# ------------------------------------ #
#             CONFIGURATION
# ------------------------------------ #

IMAGE_FOLDER  = 'imgs/inputs'
OUTPUT_DIR = 'imgs/inputs_pred'

TEXT_PROMPTS = ["drone", "uav", "flying drone", "quadcopter", "unmanned aerial vehicle"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAM3_MODEL_PATH = "models/sam3.pt"
SAM3_CONF = 0.25

MIN_BBOX_AREA = 50
PADDING = 20 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------ #
#              HELPERS
# ------------------------------------ #
def get_bboxes(mask, min_area=50):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append([x, y, x + w, y + h])
    return bboxes

def apply_sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


# ------------------------------------ #
#          PIPELINE EXECUTION
# ------------------------------------ #
print("Initializing SAM3 model...")
sam3_overrides = dict(
    conf=SAM3_CONF, 
    task="segment", 
    mode="predict", 
    model=SAM3_MODEL_PATH, 
    half=True,    # Set to False if running on CPU
    device="cuda", # Explicitly forcing GPU execution
    save=False  
)
sam3_predictor = SAM3SemanticPredictor(overrides=sam3_overrides)

# Get all image files
all_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]


print("\nStarting evaluation...")
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
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_mask_coarse.png"), coarse_mask)

    # ---------------------------------------------------------------
    # STAGE 2: Crop and Refine
    bboxes = get_bboxes(coarse_mask, min_area=MIN_BBOX_AREA)
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
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_p{b_idx}_mask.png"), current_crop_mask)

        # Map back to final global mask
        final_mask[y1_p:y2_p, x1_p:x2_p] = cv2.bitwise_or(final_mask[y1_p:y2_p, x1_p:x2_p], current_crop_mask)

    pred_mask = final_mask.copy()
    # ---------------------------------------------------------------

    print(f"Processed {idx}/{len(all_files)}: {filename}")

    # Save fine mask
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_mask_fine.png"), pred_mask)

    # if idx > 10:
    #     break