"""
train_unet.py  — Maximum CPU speed version
4 classes: 0=Road | 1=Vegetation | 2=Water | 3=Building

Speed strategy:
  - MobileNetV2 encoder: 3.4M params vs resnet18's 14.5M  (~4x faster)
  - encoder_weights=None: no download, no internet needed
  - Pre-loads all tiles into RAM tensors (zero disk I/O during training)
  - BATCH_SIZE=32 for maximum CPU throughput
  - All CPU cores used via torch.set_num_threads
  - Expected time: ~15-30 min total vs 169 min before
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TILE_SIZE  = 128
N_CLASSES  = 4
BATCH_SIZE = 32      # larger = faster on CPU
EPOCHS     = 60
LR         = 1e-3
PATIENCE   = 12
IMG_DIR    = "tiles/images"
MASK_DIR   = "tiles/masks"

# Use ALL CPU cores
n_cores = os.cpu_count() or 4
torch.set_num_threads(n_cores)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = DEVICE.type == "cuda"
print(f"Device  : {DEVICE}  |  CPU threads: {n_cores}")

# ─────────────────────────────────────────────
# Load ALL tiles into RAM as tensors once
# This eliminates all disk I/O during training
# ─────────────────────────────────────────────
print("\nLoading tiles into RAM...")
t0 = time.time()

all_imgs  = []
all_masks = []

files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".npy"))
for f in files:
    mf = f.replace("img_", "mask_")
    mp = os.path.join(MASK_DIR, mf)
    if not os.path.exists(mp):
        continue

    img  = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)
    mask = np.load(mp).astype(np.int64)

    if img.shape[1] != TILE_SIZE or mask.shape[0] != TILE_SIZE:
        continue

    if img.max() > 1.0:
        img = img / img.max()

    mask = np.clip(mask, 0, N_CLASSES - 1)
    all_imgs.append(img)
    all_masks.append(mask)

if not all_imgs:
    print("ERROR: No tiles found. Run create_tiles.py first.")
    exit()

# Stack into single tensors — loaded once, stays in RAM
imgs_tensor  = torch.tensor(np.stack(all_imgs),  dtype=torch.float32)
masks_tensor = torch.tensor(np.stack(all_masks), dtype=torch.long)

print(f"  Loaded {len(all_imgs)} tiles in {time.time()-t0:.1f}s")
print(f"  RAM used: ~{imgs_tensor.nbytes / 1e6:.0f} MB for images")

# ─────────────────────────────────────────────
# Split into train / val
# ─────────────────────────────────────────────
n_total  = len(all_imgs)
n_val    = max(2, int(0.2 * n_total))
n_train  = n_total - n_val

idx      = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
train_idx = idx[:n_train]
val_idx   = idx[n_train:]

# ─────────────────────────────────────────────
# Dataset with augmentation
# ─────────────────────────────────────────────
class FastDataset(Dataset):
    def __init__(self, imgs, masks, augment=True):
        self.imgs    = imgs      # already tensors
        self.masks   = masks
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img  = self.imgs[idx].clone()
        mask = self.masks[idx].clone()

        if self.augment:
            if torch.rand(1) > 0.5:
                img  = torch.flip(img,  dims=[2])   # horizontal flip
                mask = torch.flip(mask, dims=[1])
            if torch.rand(1) > 0.5:
                img  = torch.flip(img,  dims=[1])   # vertical flip
                mask = torch.flip(mask, dims=[0])

        return img, mask

train_ds = FastDataset(imgs_tensor[train_idx], masks_tensor[train_idx], augment=True)
val_ds   = FastDataset(imgs_tensor[val_idx],   masks_tensor[val_idx],   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=USE_GPU)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=USE_GPU)

print(f"Train: {n_train}  |  Val: {n_val}  |  Batch size: {BATCH_SIZE}")

# ─────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────
counts = torch.zeros(N_CLASSES, dtype=torch.long)
for c in range(N_CLASSES):
    counts[c] = (masks_tensor[train_idx] == c).sum()

print(f"\nClass pixel counts: { {i: int(counts[i]) for i in range(N_CLASSES)} }")
weights = counts.sum() / (N_CLASSES * (counts.float() + 1))
weights = (weights / weights.sum() * N_CLASSES).float()
print(f"Class weights     : {weights.numpy().round(3)}")

# ─────────────────────────────────────────────
# Model — MobileNetV2: 3.4M params, ~4x faster than resnet18
# encoder_weights=None means NO download, NO internet needed
# It trains from scratch but your dataset is large enough
# ─────────────────────────────────────────────
n_channels = imgs_tensor.shape[1]
print(f"\nInput channels: {n_channels}")

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,        # no download — trains faster, no internet needed
    in_channels=n_channels,
    classes=N_CLASSES,
    decoder_attention_type="scse",
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model params: {total_params:.1f}M  (vs 14.5M for resnet18)")

# ─────────────────────────────────────────────
# Loss, optimizer, scheduler
# ─────────────────────────────────────────────
ce_loss   = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=list(range(N_CLASSES)))

def loss_fn(pred, target):
    return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# ─────────────────────────────────────────────
# Quick epoch time estimate
# ─────────────────────────────────────────────
print("\nEstimating speed...")
model.train()
t0 = time.time()
_imgs, _masks = next(iter(train_loader))
_imgs, _masks = _imgs.to(DEVICE), _masks.to(DEVICE)
optimizer.zero_grad()
loss_fn(model(_imgs), _masks).backward()
optimizer.step()
batch_sec  = time.time() - t0
epoch_sec  = batch_sec * len(train_loader)
total_min  = epoch_sec * EPOCHS / 60
print(f"  ~{epoch_sec:.0f}s per epoch  (~{total_min:.0f} min for {EPOCHS} epochs)")
print(f"  Early stopping will likely finish in {total_min*0.3:.0f}–{total_min*0.6:.0f} min\n")

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
best_val   = float("inf")
no_improve = 0
times      = []

print(f"{'Ep':>4} {'Train':>8} {'Val':>8} {'LR':>9} {'Sec':>6} {'':6} ETA")
print("─" * 55)

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()

    # Train
    model.train()
    tloss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tloss += loss.item()
    tloss /= len(train_loader)

    # Validate
    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            vloss += loss_fn(model(imgs), masks).item()
    vloss /= len(val_loader)

    scheduler.step()
    elapsed = time.time() - t_ep
    times.append(elapsed)
    lr_now = scheduler.get_last_lr()[0]
    eta    = sum(times) / len(times) * (EPOCHS - epoch)
    eta_s  = f"{eta/60:.0f}m" if eta > 90 else f"{eta:.0f}s"

    is_best = vloss < best_val
    flag    = "  ← best" if is_best else ""
    print(f"{epoch:>4} {tloss:>8.4f} {vloss:>8.4f} {lr_now:>9.1e} {elapsed:>5.1f}s       ETA {eta_s}{flag}")

    if is_best:
        best_val = vloss
        no_improve = 0
        torch.save(model.state_dict(), "model.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

total = sum(times)
print(f"\nDone in {total/60:.1f} min  |  Best val: {best_val:.4f}  |  Saved → model.pth")

# ─────────────────────────────────────────────
# Per-class IoU
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

tp = np.zeros(N_CLASSES)
fp = np.zeros(N_CLASSES)
fn = np.zeros(N_CLASSES)

with torch.no_grad():
    for imgs, masks in val_loader:
        preds = torch.argmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy()
        gt    = masks.numpy()
        for c in range(N_CLASSES):
            tp[c] += ((preds == c) & (gt == c)).sum()
            fp[c] += ((preds == c) & (gt != c)).sum()
            fn[c] += ((preds != c) & (gt == c)).sum()

names = ["road", "vegetation", "water", "building"]
ious  = tp / (tp + fp + fn + 1e-6)
print(f"\n{'Class':<15} {'IoU':>6}")
print("─" * 22)
for c in range(N_CLASSES):
    bar = "█" * int(ious[c] * 20)
    print(f"{names[c]:<15} {ious[c]:>6.3f}  {bar}")
print(f"\nmIoU: {ious.mean():.3f}")