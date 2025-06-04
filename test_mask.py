import os
import random
from PIL import Image
import numpy as np

# === YOUR MASK FOLDER PATH ===
mask_dir = "C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/masks"

# === List all mask files ===
sample_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')]

# === Randomly select 5 files ===
#sample_files = random.sample(mask_files, 5)

print("------ RANDOM MASK PIXEL CHECK ------")

for filename in sample_files:
    mask_path = os.path.join(mask_dir, filename)
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    nonzero_pixels = np.sum(mask_np > 0)
    total_pixels = mask_np.size

    print(f"{filename}: {nonzero_pixels} / {total_pixels} pixels > 0")

print("--------------------------------------")
