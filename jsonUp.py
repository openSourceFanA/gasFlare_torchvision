import json
import os

json_path = "C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/Gas-flare-monitoring.v2i.png-mask-semantic/Ttest/_annotations.coco.json"
output_path = "C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/Gas-flare-monitoring.v2i.png-mask-semantic/Ttest/_annotations_fixed.coco.json"


# ==== PATHS YOU MUST EDIT ====
#json_path = "C:/your/path/to/train/_annotations.coco.json"  # <-- put your real JSON path here
#output_path = "C:/your/path/to/train/_annotations_fixed.coco.json"  # <-- save to new file
# =============================

# Load the original JSON
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Fix filenames inside "images"
for img in coco_data['images']:
    old_name = img['file_name']
    
    # Remove weird part and fix extension
    if "_jpg.rf." in old_name:
        base = old_name.split("_jpg.rf.")[0]
        img['file_name'] = f"{base}.jpg"
    else:
        print(f"[Warning] Skipped: {old_name}")

# Save the fixed JSON
with open(output_path, 'w') as f:
    json.dump(coco_data, f)

print("✅ JSON file fixed and saved to:", output_path)
