import os
import cv2
import time
import torch
import argparse
import numpy as np

from nets import utils
from nets.ipsseg import IPSSeg
from thop import profile


# ------------------------------------ #
#             CONFIGURATION
# ------------------------------------ #

# Parse arguments
parser = argparse.ArgumentParser(description="Argument parser for IPS-Seg.")
parser.add_argument("--in_dir", type=str, default='imgs/inputs', help="Input folder")
parser.add_argument("--out_dir", type=str, default='imgs/outputs_ipsseg', help="Output folder")
parser.add_argument("--model_path", type=str, default='models/ipsseg_2s.pth', help="Model path")
args = parser.parse_args()


IMAGE_FOLDER  = args.in_dir
OUTPUT_DIR = args.out_dir
MODEL_PATH = args.model_path

IMG_SIZE = 256
MIN_BBOX_AREA = 50
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------ #
#               EXECUTION
# ------------------------------------ #
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nLoading model using {device}: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
# print(f"Original keys: {len(state_dict.keys())}, Filtered keys: {len(filtered_state_dict.keys())}")

unet = IPSSeg(3,1)
unet.load_state_dict(filtered_state_dict, strict=False)
unet.to(device)
unet.eval()
print("Model loaded successfully!")


# Count model parameters and FLOPs
input_size=(1, 3, IMG_SIZE, IMG_SIZE)
dummy_input = torch.randn(*input_size).to(device)
flops, params = profile(unet, inputs=(dummy_input,), verbose=False)
print(f"Rarams: {params:,}")
print(f"FLOPs: {flops:,}")


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]


print(f"\nStarting inference: {IMAGE_FOLDER}")
for idx, filename in enumerate(all_files, 1):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    filename_mask = filename.replace('jpg', 'png')
    fname = os.path.splitext(filename)[0]
    
    # Load Image
    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]

    # ---------------------------------------------------------------
    # STAGE 1: IPSSeg Inference
    input_img = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    input_img = np.array([input_img])
    input_img = torch.Tensor(input_img).permute(0, 3, 1, 2) / 255  # Convert from (N, H, W, C) to (N, C, H, W)
    input_img = input_img.to(device)

    pred = unet(input_img)
    pred = torch.sigmoid(pred)
    pred_np = pred.cpu().detach().numpy().squeeze()
    pred_np = (pred_np*255).astype(np.uint8)
    pred_np = cv2.resize(pred_np, (w_img, h_img))
    coarse_mask = pred_np.copy()

    # Save coarse mask
    final_coarse_mask = np.where(coarse_mask > 128, 255, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_ipsseg.png"), final_coarse_mask)

    # ---------------------------------------------------------------
    # STAGE 2: Crop and Refine
    bboxes = utils.get_bboxes(coarse_mask, min_area=MIN_BBOX_AREA)
    final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    PADDING = 20

    for b_idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        x1_p, y1_p = max(0, x1 - PADDING), max(0, y1 - PADDING)
        x2_p, y2_p = min(w_img, x2 + PADDING), min(h_img, y2 + PADDING)

        crop = image[y1_p:y2_p, x1_p:x2_p].copy()
        if crop.size == 0: continue

        input_img = cv2.resize(crop, (IMG_SIZE,IMG_SIZE))
        input_img = np.array([input_img])
        input_img = torch.Tensor(input_img).permute(0, 3, 1, 2) / 255  # Convert from (N, H, W, C) to (N, C, H, W)
        input_img = input_img.to(device)

        pred = unet(input_img)
        pred = torch.sigmoid(pred)
        pred_np = pred.cpu().detach().numpy().squeeze()
        pred_np = (pred_np*255).astype(np.uint8)
        pred_np = cv2.resize(pred_np, (x2_p-x1_p, y2_p-y1_p))

        final_mask[y1_p:y2_p, x1_p:x2_p] = cv2.bitwise_or(final_mask[y1_p:y2_p, x1_p:x2_p], pred_np)


    pred_mask = np.where(final_mask > 128, 255, 0).astype(np.uint8)
    # ---------------------------------------------------------------

    pred_path = os.path.join(OUTPUT_DIR, f'{fname}_ipsseg2s.png')
    cv2.imwrite(pred_path, pred_mask)

    print(f"Processed {idx}/{len(all_files)}: {filename}")

print(f'Outputs saved to {OUTPUT_DIR}')