import os
import torch
from PIL import Image
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import numpy as np

class CustomCocoDataset(CocoDetection):
    def __init__(self, img_dir, mask_dir, ann_file, transforms=None):
        super().__init__(img_dir, ann_file)
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        # Load image and original COCO target
        img, coco_target = super().__getitem__(idx)
        img_id = self.ids[idx]
        #ann_ids = self.coco.getAnnIds(imgIds=img_id)
        #anns = self.coco.loadAnns(ann_ids)

        # Get image filename
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        mask_filename = file_name.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load and process mask
        full_mask = Image.open(mask_path).convert("L")
        full_mask = torch.tensor(np.array(full_mask) > 0).float()  # shape [H, W]

        # Convert image to tensor
        img = F.to_tensor(img)

        # Parse boxes and labels
        boxes = []
        labels = []
        for obj in coco_target:
            bbox = obj["bbox"]
            x, y, w, h = bbox
            x1, y1, x2, y2 = map(int, [x, y, x + w, y + h])
            boxes.append([x1, y1, x2, y2])
            labels.append(obj["category_id"] - 1)  # adjust if needed

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Generate one mask per object based on bounding box
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, full_mask.shape[1])
            y2 = min(y2, full_mask.shape[0])
            crop = full_mask[y1:y2, x1:x2]
            padded = torch.zeros_like(full_mask)
            padded[y1:y2, x1:x2] = crop
            masks.append(padded)

        if len(masks) == 0:
            masks = torch.zeros((0, *full_mask.shape), dtype=torch.uint8)
        else:
            masks = torch.stack(masks).float()

        # Final target dict
        new_target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        if self.transforms:
            img, new_target = self.transforms(img, new_target)

        return img, new_target
        """
        mask = Image.open(mask_path).convert("L")
        mask = torch.tensor(np.array(mask) > 0).float().clamp(0, 1)  # binary mask
        #masks = mask.unsqueeze(0)  # shape: [1, H, W] (if you only have one object per image)
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = mask[y1:y2, x1:x2]
            # Pad cropped mask back to original size (simplest fix)
            full_mask = torch.zeros_like(mask)
            full_mask[y1:y2, x1:x2] = cropped
            masks.append(full_mask)
        # Stack all object masks
        masks = torch.stack(masks)
        
        # Collect boxes and labels from annotations
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] - 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        # Convert image to tensor
        img = F.to_tensor(img)

        print("Labels in batch:", labels)
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                print("Invalid box found:", box)


        # Apply transforms if available
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
        """
