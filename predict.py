"""
predict.py
Matches osm_mask.py: 4 classes  0=Road | 1=Vegetation | 2=Water | 3=Building

Auto-detects decoder_attention_type from saved model.pth weights
so it always loads correctly regardless of how train_unet.py was run.
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# Config — must match osm_mask.py
# ─────────────────────────────────────────────
MODEL_PATH = "model.pth"
TILE_DIR   = "tiles/images"
N_CLASSES  = 4          # 0=Road 1=Vegetation 2=Water 3=Building
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES  = ["Road", "Vegetation", "Water", "Building"]
CLASS_COLORS = {
    0: [220, 80,  80],    # road        — red
    1: [60,  180, 75],    # vegetation  — green
    2: [66,  133, 244],   # water       — blue
    3: [255, 220, 80],    # building    — yellow
}

# ─────────────────────────────────────────────
# Step 1: Detect input channels from saved tiles
# ─────────────────────────────────────────────
sample_files = [f for f in sorted(os.listdir(TILE_DIR)) if f.endswith(".npy")]
if not sample_files:
    raise FileNotFoundError("No tiles found in tiles/images/. Run create_tiles.py first.")

n_channels = np.load(os.path.join(TILE_DIR, sample_files[0])).shape[0]
print(f"Input channels: {n_channels}")

# ─────────────────────────────────────────────
# Step 2: Inspect saved weights → auto-detect architecture
#
# This prevents the RuntimeError about missing/unexpected scse keys.
# Reads the .pth file first and checks whether attention layers exist,
# then builds the model to exactly match the saved weights.
# ─────────────────────────────────────────────
print(f"Reading {MODEL_PATH}...")
state = torch.load(MODEL_PATH, map_location=DEVICE)

has_scse       = any("attention1.attention" in k for k in state.keys())
attention_type = "scse" if has_scse else None
print(f"  Detected decoder_attention_type = {attention_type!r}")

# ─────────────────────────────────────────────
# Step 3: Build model matching saved weights exactly
# ─────────────────────────────────────────────
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=n_channels,
    classes=N_CLASSES,                      # 4 — matches osm_mask.py
    decoder_attention_type=attention_type,  # auto-matched — never hardcode this
)

model.load_state_dict(state, strict=True)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded successfully on {DEVICE}\n")

# ─────────────────────────────────────────────
# Step 4: Predict on tiles
# ─────────────────────────────────────────────
def colorize(mask):
    """Convert (H,W) class-index mask → (H,W,3) RGB image."""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color[mask == cls] = rgb
    return color

def predict_tile(img_arr):
    """img_arr: (C,H,W) float32 normalized [0,1]."""
    t = torch.tensor(img_arr).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        return torch.argmax(model(t), dim=1).squeeze().cpu().numpy()

N_SHOW   = min(9, len(sample_files))
preds    = []
overlays = []

for fname in sample_files[:N_SHOW]:
    img_arr = np.load(os.path.join(TILE_DIR, fname)).astype(np.float32)

    # Safety: re-normalize if tiles were saved un-normalized (values > 1)
    if img_arr.max() > 1.0:
        print(f"  Re-normalizing {fname} (max={img_arr.max():.0f})")
        p2  = np.percentile(img_arr,  2, axis=(1, 2), keepdims=True)
        p98 = np.percentile(img_arr, 98, axis=(1, 2), keepdims=True)
        img_arr = np.clip((img_arr - p2) / (p98 - p2 + 1e-6), 0, 1)

    pred = predict_tile(img_arr)
    preds.append(pred)

    # RGB preview — bands 2,1,0 = Red,Green,Blue for standard Sentinel-2
    try:
        rgb = np.clip(np.stack([img_arr[2], img_arr[1], img_arr[0]], axis=-1), 0, 1)
    except IndexError:
        rgb = np.zeros((TILE_DIR, 128, 3), dtype=np.float32)

    overlays.append((rgb, colorize(pred)))

# ─────────────────────────────────────────────
# Step 5: Plot RGB | Prediction side-by-side grid
# ─────────────────────────────────────────────
n_cols = 3
n_rows = int(np.ceil(N_SHOW / n_cols))
fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 4, n_rows * 8))
axes = axes.reshape(n_rows * 2, n_cols)

for i, (rgb, colored) in enumerate(overlays):
    r, c = (i // n_cols) * 2, i % n_cols
    axes[r  ][c].imshow(rgb);     axes[r  ][c].set_title(f"Tile {i} — RGB",        fontsize=9); axes[r  ][c].axis("off")
    axes[r+1][c].imshow(colored); axes[r+1][c].set_title(f"Tile {i} — Prediction", fontsize=9); axes[r+1][c].axis("off")

# Hide empty axes
for j in range(N_SHOW, n_rows * n_cols):
    axes[(j//n_cols)*2  ][j%n_cols].axis("off")
    axes[(j//n_cols)*2+1][j%n_cols].axis("off")

# Legend
patches = [mpatches.Patch(color=[v/255 for v in rgb], label=name)
           for name, rgb in zip(CLASS_NAMES, CLASS_COLORS.values())]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle("Segmentation Predictions  (0=Road | 1=Vegetation | 2=Water | 3=Building)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("prediction_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved prediction_output.png")

# ─────────────────────────────────────────────
# Step 6: Class distribution across predicted tiles
# ─────────────────────────────────────────────
print("\nClass distribution in predictions:")
total = sum(p.size for p in preds)
for c, name in enumerate(CLASS_NAMES):
    count = sum((p == c).sum() for p in preds)
    bar   = "█" * int(count / total * 40)
    print(f"  {name:<15}: {count/total*100:5.1f}%  {bar}")

print("\nHealthy range: Road 20-50%, Vegetation 20-50%, Water 0-20%, Building 10-40%")
print("If one class > 80%: re-run osm_mask.py, check mask_preview.png, then retrain")