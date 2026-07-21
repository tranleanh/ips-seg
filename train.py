import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from glob import glob
from skimage import io
from thop import profile
from tqdm.auto import tqdm
from nets.ipsseg import IPSSeg
from skimage.transform import resize


# Configurations
saved_mode_name = f'IPSSeg'
SHOW_DATA_SAMPLE = False
SAVE_GAP = 5
RESUME = False

# Instantiate model
unet = IPSSeg(3,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining model using {device}")

if RESUME:
    state_dict = torch.load("checkpoints/path_to_checkpoints.pth", map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
    # print(f"Original keys: {len(state_dict.keys())}, Filtered keys: {len(filtered_state_dict.keys())}")
    unet.load_state_dict(filtered_state_dict, strict=False)

unet.to(device)

# Load dataset
volumes_path = 'path_to_imgs'
labels_path = 'path_to_masks'

# Get sorted list of image and mask files
volume_files = sorted(glob(os.path.join(volumes_path, '*')))
label_files = sorted(glob(os.path.join(labels_path, '*')))

# Read images into numpy arrays
img_size_in = 256
img_size_out = img_size_in

volumes = []
for f in tqdm(volume_files):
  volumes.append(resize(io.imread(f), (img_size_in, img_size_in), preserve_range=True, anti_aliasing=True).astype(np.uint8))
volumes = np.array(volumes)

labels = []
for f in tqdm(label_files):
  labels.append(resize(io.imread(f), (img_size_out, img_size_out), preserve_range=True, anti_aliasing=True).astype(np.uint8))
labels = np.array(labels)

print("volumes shape:", volumes.shape)
print("labels shape:", labels.shape)

# Display one data sample
if SHOW_DATA_SAMPLE:
    id_ = random.randint(0,volumes.shape[0]-1)
    # print(id_)

    # Normalize images to [0, 1] for blending
    vol_img = volumes[id_].astype(np.float32) / 255.0
    lbl_img = labels[id_].astype(np.float32) / 255.0

    # If label is single channel, convert to 3 channels for blending
    if lbl_img.ndim == 2:
        lbl_img_color = np.stack([lbl_img, np.zeros_like(lbl_img), np.zeros_like(lbl_img)], axis=-1)
    else:
        lbl_img_color = lbl_img

    # If input is grayscale, convert to 3 channels for blending
    if vol_img.ndim == 2:
        vol_img_color = np.stack([vol_img, vol_img, vol_img], axis=-1)
    else:
        vol_img_color = vol_img

    # Blend images: overlay label in red on top of the input image
    alpha = 0.5
    blended = (1 - alpha) * vol_img_color + alpha * lbl_img_color

    # Display
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1), plt.title("Input"), plt.imshow(vol_img, cmap="gray"), plt.axis("off")
    plt.subplot(1, 3, 2), plt.title("Label"), plt.imshow(lbl_img, cmap="gray"), plt.axis("off")
    plt.subplot(1, 3, 3), plt.title("Blended"), plt.imshow(blended), plt.axis("off")
    plt.tight_layout()
    plt.savefig('sample_data_visualization.png', dpi=150, bbox_inches='tight')
    print("Sample data visualization saved as 'sample_data_visualization.png'")
    plt.close()

# Prepare dataset
volumes = torch.Tensor(volumes).permute(0, 3, 1, 2)/255  # (N, H, W, C) -> (N, C, H, W)
labels = torch.Tensor(labels)[:, None, :, :]/255  # Add channel dimension for labels
dataset = torch.utils.data.TensorDataset(volumes, labels)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# Count model parameters and FLOPs
input_size=(1, 3, img_size_in, img_size_in)
dummy_input = torch.randn(*input_size).to(device)
flops, params = profile(unet, inputs=(dummy_input,), verbose=False)
print(f"Number of parameters: {params:,}")
print(f"Number of FLOPs (Multiply-Adds): {flops:,}")
print(f"Number of FLOPs (GFLOPs): {flops/1e9:.2f} GFLOPs")

# Define criterion
criterion = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = optim.Adam(unet.parameters(), lr=2e-4)

# Train
n_epochs = 50

for epoch in range(n_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for mini_images, mini_labels in tqdm(dataloader):
        mini_images = mini_images.to(device)
        mini_labels = mini_labels.to(device)
        preds = unet(mini_images)
        loss = criterion(preds, mini_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"\nEpoch: {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

    # Save model
    if (epoch+1) % SAVE_GAP == 0:
        torch.save(unet.state_dict(), f'checkpoints/{saved_mode_name}_e{epoch+1}.pth')