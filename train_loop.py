from dataset_loader import get_dataset
from model_setup import get_model_instance_segmentation
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

# ==== Set dataset paths ====
train_img_dir = "C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/imgs"
train_mask_dir = "C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/masks"
train_ann_file = "C:/Users/Amirhossein/AppData/Local/Programs/Python/Python311/Lib/site-packages/P_exercise/gasFlare/gas3_torchvision/ann/_annotations_fixed_train.coco.json"

sample_files = [f for f in os.listdir(train_mask_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Load dataset ====
train_dataset = get_dataset(train_img_dir, train_mask_dir, train_ann_file)

print('1')

# Custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

print('2')

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

print('3')

# =Load model
model = get_model_instance_segmentation(num_classes=2)  # Background + flare
if (torch.cuda.is_available()):
    print("cuda is availabel :)")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('4')

# == optimizer 
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
print('5')

# =training loop (simple version) ====
model.train()

start_time_T = time.time()

for epoch in range(10):
    start_time = time.time()
    #batch_idx = 0

    total_loss = 0
    print('6')

    for batch_idx, (images, targets) in enumerate(train_loader):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                print(f" NaN or Inf in loss at batch {batch_idx}")
                print("Loss dict:", loss_dict)
                print("Sample target:", targets[0])
                continue  # Skip this batch

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += losses.item()

            print("Labels in batch:", targets[0]['labels'])
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} Loss: {losses.item():.4f}")

        except Exception as e:
            print(f" Exception in batch {batch_idx}: {e}")
            continue
    
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"EPOCH:{epoch}>>Training took {hours:02d}:{minutes:02d}:{seconds:02d}")

    if epoch%5 == 0:
        torch.save(model.state_dict(), "model.path")
        print(f"MODEL SAVED! ep = {epoch}")


end_time_T = time.time()
total_seconds = int(end_time_T - start_time_T)
hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60
print(f"TOTAL:: >>Training took {hours:02d}:{minutes:02d}:{seconds:02d}")

#torch.save(model.state_dict(), "model10.pth")

"""
    for (images, targets) in train_loader:
        #print('7')
        batch_idx += 1

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if batch_idx == 0:  # Only for the first batch
            print("Image shape:", images[0].shape)
            print("Target keys:", targets[0].keys())
            print("Target sample:", targets[0])

            # Visualize the first mask
            mask = targets[0]['masks'][0].cpu().numpy()  # Extract the first mask
            plt.imshow(mask, cmap='gray')  # Show the mask in grayscale
            plt.show()
            exit()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += losses.item()

        if torch.isnan(losses) or torch.isinf(losses):
            print("⚠️ NaN or Inf detected in loss!")
            exit()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} Loss: {losses.item():.4f}")

    print(f"Epoch {epoch+1} - Total Loss: {total_loss:.4f}")
"""
