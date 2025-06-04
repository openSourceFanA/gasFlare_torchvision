import os
import shutil

# Current folder where mixed images+masks are
input_folder = r"C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/Gas-flare-monitoring.v2i.png-mask-semantic/Ttest"  # <--- change this

# New folders
output_imgs_folder = os.path.join(input_folder, "Timg")
output_masks_folder = os.path.join(input_folder, "Tmask")

# Make output folders if they don't exist
os.makedirs(output_imgs_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)

# Go through all files
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") and "_mask" not in filename:
        # It's an image
        new_name = filename.split('_jpg.rf')[0] + ".jpg"
        shutil.copy(os.path.join(input_folder, filename), os.path.join(output_imgs_folder, new_name))

    elif filename.endswith("_mask.png"):
        # It's a mask
        new_name = filename.split('_jpg.rf')[0] + "_mask.png"
        shutil.copy(os.path.join(input_folder, filename), os.path.join(output_masks_folder, new_name))

print("✅ Files renamed and moved successfully.")
