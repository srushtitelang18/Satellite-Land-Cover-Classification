"""
train.py  — Alternative to UNet
===============================
Uses THREE models, ranked by speed vs accuracy:

  OPTION A: DeepLabV3+ with EfficientNet-B0 encoder
            → Better accuracy than UNet on satellite imagery
            → Atrous convolutions capture multi-scale features
            → ~8M params (vs UNet resnet18's 14.5M)
            → ~30-50 min on CPU

  OPTION B: FPN (Feature Pyramid Network) with MobileNetV2
            → Excellent for multi-scale detection (roads + buildings together)
            → Only 5M params
            → ~15-25 min on CPU  ← RECOMMENDED for CPU

  OPTION C: Random Forest on spectral features (NO deep learning)
            → Trains in < 5 minutes on CPU, no GPU needed
            → Decent accuracy using NDVI/NDWI/NDBI + raw bands
            → Best choice if you want results immediately

Set MODEL_CHOICE below to "FPN", "DeepLabV3+", or "RandomForest"
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# ★ CHANGE THIS to switch models ★
# Options: "FPN"  |  "DeepLabV3+"  |  "RandomForest"
MODEL_CHOICE = "FPN"
# ─────────────────────────────────────────────

TILE_SIZE  = 128
N_CLASSES  = 4       # 0=Road 1=Vegetation 2=Water 3=Building
BATCH_SIZE = 32
EPOCHS     = 60
LR         = 1e-3
PATIENCE   = 12
IMG_DIR    = "tiles/images"
MASK_DIR   = "tiles/masks"

CLASS_NAMES  = ["Road", "Vegetation", "Water", "Building"]
CLASS_COLORS = {0:[220,80,80], 1:[60,180,75], 2:[66,133,244], 3:[255,220,80]}

n_cores = os.cpu_count() or 4
torch.set_num_threads(n_cores)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = DEVICE.type == "cuda"
print(f"Model   : {MODEL_CHOICE}")
print(f"Device  : {DEVICE}  |  Threads: {n_cores}")

# ════════════════════════════════════════════════════════════
# OPTION C — Random Forest (no deep learning, fastest)
# ════════════════════════════════════════════════════════════
if MODEL_CHOICE == "RandomForest":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics  import classification_report
    import joblib

    print("\nLoading tiles for Random Forest...")
    X_list, y_list = [], []

    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".npy"))
    for f in files:
        mf = f.replace("img_", "mask_")
        mp = os.path.join(MASK_DIR, mf)
        if not os.path.exists(mp): continue

        img  = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)   # (C, H, W)
        mask = np.load(mp).astype(np.int64).flatten()                  # (H*W,)

        C, H, W = img.shape

        # Feature engineering: raw bands + spectral indices
        # Flatten spatial dims → (H*W, C)
        flat = img.reshape(C, -1).T   # (H*W, C)

        # Compute spectral indices as extra features
        eps   = 1e-6
        red   = img[2].flatten()
        nir   = img[3].flatten() if C > 3 else img[0].flatten()
        green = img[1].flatten()
        ndvi  = (nir - red)   / (nir   + red   + eps)
        ndwi  = (green - nir) / (green + nir   + eps)
        ndbi  = (red - nir)   / (red   + nir   + eps)   # proxy without SWIR

        # Stack raw bands + indices → (H*W, C+3)
        features = np.hstack([flat, ndvi[:,None], ndwi[:,None], ndbi[:,None]])

        X_list.append(features)
        y_list.append(mask)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    y = np.clip(y, 0, N_CLASSES - 1)

    print(f"Feature matrix: {X.shape}  |  Labels: {y.shape}")

    # Subsample for speed (RF on 10M+ pixels is slow)
    MAX_SAMPLES = 500_000
    if len(y) > MAX_SAMPLES:
        idx = np.random.choice(len(y), MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]
        print(f"Subsampled to {MAX_SAMPLES} pixels for speed")

    # Class-balanced split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,               # use all CPU cores
        random_state=42,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training done in {train_time:.0f}s")

    # Evaluate
    y_pred = rf.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

    # Save
    joblib.dump(rf, "model_rf.pkl")
    print("Saved model_rf.pkl")
    print("Run predict.py to generate predictions.")
    exit()


# ════════════════════════════════════════════════════════════
# OPTION A / B — Deep learning (FPN or DeepLabV3+)
# ════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# Load all tiles into RAM tensors (fast training)
# ─────────────────────────────────────────────
print("\nLoading tiles into RAM...")
t0 = time.time()
all_imgs, all_masks = [], []

for f in sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".npy")):
    mf = f.replace("img_", "mask_")
    mp = os.path.join(MASK_DIR, mf)
    if not os.path.exists(mp): continue

    img  = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)
    mask = np.load(mp).astype(np.int64)

    if img.shape[1] != TILE_SIZE or mask.shape[0] != TILE_SIZE: continue
    if img.max() > 1.0: img = img / img.max()

    mask = np.clip(mask, 0, N_CLASSES - 1)
    all_imgs.append(img)
    all_masks.append(mask)

if not all_imgs:
    print("ERROR: No tiles found. Run create_tiles.py first.")
    exit()

imgs_t  = torch.tensor(np.stack(all_imgs),  dtype=torch.float32)
masks_t = torch.tensor(np.stack(all_masks), dtype=torch.long)
print(f"  {len(all_imgs)} tiles in {time.time()-t0:.1f}s  |  RAM: ~{imgs_t.nbytes/1e6:.0f}MB")

# ─────────────────────────────────────────────
# Dataset with augmentation
# ─────────────────────────────────────────────
class TileDataset(Dataset):
    def __init__(self, imgs, masks, augment=False):
        self.imgs    = imgs
        self.masks   = masks
        self.augment = augment

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img  = self.imgs[idx].clone()
        mask = self.masks[idx].clone()
        if self.augment:
            if torch.rand(1) > 0.5: img = torch.flip(img,  [2]); mask = torch.flip(mask, [1])
            if torch.rand(1) > 0.5: img = torch.flip(img,  [1]); mask = torch.flip(mask, [0])
            # Random brightness jitter (satellite-specific)
            if torch.rand(1) > 0.5: img = img * (0.8 + torch.rand(1) * 0.4)
            img = img.clamp(0, 1)
        return img, mask

n       = len(all_imgs)
n_val   = max(2, int(0.2 * n))
idx     = torch.randperm(n, generator=torch.Generator().manual_seed(42))
t_idx   = idx[:n-n_val]
v_idx   = idx[n-n_val:]

train_ds = TileDataset(imgs_t[t_idx], masks_t[t_idx], augment=True)
val_ds   = TileDataset(imgs_t[v_idx], masks_t[v_idx], augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=USE_GPU)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=USE_GPU)
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Batch: {BATCH_SIZE}")

# ─────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────
counts = torch.zeros(N_CLASSES, dtype=torch.long)
for c in range(N_CLASSES):
    counts[c] = (masks_t[t_idx] == c).sum()
weights = counts.sum() / (N_CLASSES * (counts.float() + 1))
weights = (weights / weights.sum() * N_CLASSES).float()
print(f"Class weights: {weights.numpy().round(3)}")

# ─────────────────────────────────────────────
# Build chosen model
# ─────────────────────────────────────────────
n_channels = imgs_t.shape[1]
print(f"Input channels: {n_channels}")

if MODEL_CHOICE == "FPN":
    # Feature Pyramid Network
    # Great for detecting objects at multiple scales simultaneously
    # Roads (thin, small) and buildings (large) benefit from this
    model = smp.FPN(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=n_channels,
        classes=N_CLASSES,
        upsampling=4,
    )
    print("Architecture: FPN + MobileNetV2")

elif MODEL_CHOICE == "DeepLabV3+":
    # DeepLabV3+ with atrous (dilated) convolutions
    # Captures context at multiple scales without losing resolution
    # Best accuracy for semantic segmentation tasks
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=n_channels,
        classes=N_CLASSES,
    )
    print("Architecture: DeepLabV3+ + EfficientNet-B0")

else:
    raise ValueError(f"Unknown MODEL_CHOICE: {MODEL_CHOICE}")

model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters: {total_params:.1f}M")

# ─────────────────────────────────────────────
# Loss: CrossEntropy + Dice + Focal
# Focal loss specifically targets hard-to-classify pixels (roads, water)
# ─────────────────────────────────────────────
ce_loss    = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
dice_loss  = smp.losses.DiceLoss(mode="multiclass")
focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)

def loss_fn(pred, target):
    # 40% CE + 40% Dice + 20% Focal
    return (0.4 * ce_loss(pred, target) +
            0.4 * dice_loss(pred, target) +
            0.2 * focal_loss(pred, target))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,          # warm-up for 10% of training
    div_factor=10,
    final_div_factor=100,
)

# ─────────────────────────────────────────────
# Estimate speed
# ─────────────────────────────────────────────
print("\nEstimating speed...")
model.train()
t0 = time.time()
_i, _m = next(iter(train_loader))
_i, _m = _i.to(DEVICE), _m.to(DEVICE)
optimizer.zero_grad()
loss_fn(model(_i), _m).backward()
optimizer.step()
epoch_sec = (time.time() - t0) * len(train_loader)
print(f"  ~{epoch_sec:.0f}s/epoch  (~{epoch_sec*EPOCHS/60:.0f} min total)")
print()

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
best_val   = float("inf")
no_improve = 0
times      = []

print(f"{'Ep':>4} {'Train':>8} {'Val':>8} {'Sec':>6}  ETA")
print("─" * 45)

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()

    model.train()
    tloss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tloss += loss.item()
    tloss /= len(train_loader)

    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            vloss += loss_fn(model(imgs), masks).item()
    vloss /= len(val_loader)

    elapsed = time.time() - t_ep
    times.append(elapsed)
    eta = sum(times)/len(times) * (EPOCHS - epoch)
    eta_s = f"{eta/60:.0f}m" if eta > 90 else f"{eta:.0f}s"

    is_best = vloss < best_val
    flag    = "  ← BEST" if is_best else ""
    print(f"{epoch:>4} {tloss:>8.4f} {vloss:>8.4f} {elapsed:>5.1f}s  ETA {eta_s}{flag}")

    if is_best:
        best_val = vloss
        no_improve = 0
        torch.save(model.state_dict(), "model.pth")
        # Save architecture info so predict.py loads correctly
        torch.save({"model_choice": MODEL_CHOICE, "n_channels": n_channels,
                    "n_classes": N_CLASSES}, "model_config.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nDone in {sum(times)/60:.1f} min  |  Best val: {best_val:.4f}  →  model.pth")

# ─────────────────────────────────────────────
# Per-class IoU
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()
tp = fp = fn = np.zeros(N_CLASSES)
tp = np.zeros(N_CLASSES); fp = np.zeros(N_CLASSES); fn = np.zeros(N_CLASSES)

with torch.no_grad():
    for imgs, masks in val_loader:
        preds = torch.argmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy()
        gt    = masks.numpy()
        for c in range(N_CLASSES):
            tp[c] += ((preds==c)&(gt==c)).sum()
            fp[c] += ((preds==c)&(gt!=c)).sum()
            fn[c] += ((preds!=c)&(gt==c)).sum()

ious = tp / (tp + fp + fn + 1e-6)
print(f"\n{'Class':<15} {'IoU':>6}  Bar")
print("─" * 40)
for c in range(N_CLASSES):
    bar = "█" * int(ious[c] * 25)
    print(f"{CLASS_NAMES[c]:<15} {ious[c]:>6.3f}  {bar}")
print(f"\nmIoU: {ious.mean():.3f}")