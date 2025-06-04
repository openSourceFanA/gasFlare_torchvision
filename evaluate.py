# evaluate.py
import os, cv2, torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from model_setup import get_model_instance_segmentation
from custom_dataset import CustomCocoDataset   # same class you used for training
import matplotlib.pyplot as plt

# -------- paths -------------------------------------------------------------
weights_path   = "model10.pth"                              # <- model weights
val_img_dir    = r"C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/test_imgs"                                          # <- images to test
val_mask_dir   = r"C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/test_masks"                  # <- optional GT masks
val_json_path  = r"C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/ann/_annotations_fixed_test.coco.json"
out_dir        = "outputs10"                                # <- where to write results
os.makedirs(out_dir, exist_ok=True)

# -------- model -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = get_model_instance_segmentation(num_classes=2)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval().to(device)

# -------- data loader (no batching for simplicity) --------------------------
val_dataset = CustomCocoDataset(val_img_dir, val_mask_dir, val_json_path)
to_tensor   = transforms.ToTensor()

# -------- helpers -----------------------------------------------------------
def draw_masks(img_tensor, masks, color=(0,255,0)):
    """Overlay all masks (binary [H,W]) onto cv2 image."""
    img = np.ascontiguousarray(np.array(to_pil_image(img_tensor))[:, :, ::-1])

    for m in masks:
        m = (m > 0.5).cpu().numpy().astype(np.uint8)*255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 2)
    return img

def iou(pred, gt):
    """IoU of two binary masks."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / union if union else 0

# -------- evaluation loop ---------------------------------------------------
total_det = 0
conf_sum  = 0
ious      = []

with torch.no_grad():
    for idx in range(len(val_dataset)):
        img, target = val_dataset[idx]
        pred = model([img.to(device)])[0]

        # --- keep predictions with score > 0.5
        keep = pred['scores'] > 0.1
        boxes = pred['boxes'][keep]
        masks = pred['masks'][keep].squeeze(1)         # -> [N,H,W]
        scores= pred['scores'][keep]

        total_det += len(scores)
        conf_sum  += scores.sum().item()

        # --- IoU per object (if GT masks exist)
        img_name = val_dataset.coco.loadImgs(val_dataset.ids[idx])[0]['file_name']
        gt_mask_path = os.path.join(val_mask_dir, img_name.replace('.jpg','_mask.png'))
        if os.path.exists(gt_mask_path):
            gt = torch.tensor(np.array(Image.open(gt_mask_path)) > 0)
            if len(masks):
                best = max(iou((m>0.5).cpu(), gt) for m in masks)
                ious.append(best)

        # --- visualisation ---------------------------------------------------
        img_bgr = draw_masks(img, masks)
        cv2.imwrite(os.path.join(out_dir, f"pred_{img_name}"), img_bgr)

        if idx < 3:       # show a few samples in a pop‑up
            plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            plt.title(f"{img_name} – detections: {len(scores)}")
            plt.axis('off')
            plt.show()

# -------- simple metrics -----------------------------------------------------
n_imgs = len(val_dataset)
mean_det = total_det / n_imgs
mean_conf= conf_sum  / total_det if total_det else 0
mean_iou = np.mean(ious) if ious else None

print("\n=== Simple evaluation (epoch‑1 quick check) ===")
print(f"Images evaluated      : {n_imgs}")
print(f"Avg detections / image: {mean_det:.2f}")
print(f"Mean confidence       : {mean_conf:.3f}")
if mean_iou is not None:
    print(f"Mean IoU (GT masks)   : {mean_iou:.3f}")
else:
    print("Mean IoU              : GT masks not supplied")
print("Visual results saved to:", out_dir)
