"""
create_tiles.py
Matches osm_mask.py: 4 classes  0=Road | 1=Vegetation | 2=Water | 3=Building
"""

import rasterio
import numpy as np
import os

# ─────────────────────────────────────────────
# Config — must match your image & mask files
# ─────────────────────────────────────────────
IMAGE_PATH = "Sentinel_AI_Project.tif"
MASK_PATH  = "training_mask_osm.tif"
TILE_SIZE  = 128
STRIDE     = 64       # overlap = TILE_SIZE - STRIDE; 64 gives 2× more tiles
MIN_NONROAD_PX = 5    # skip tiles that are almost entirely road (class 0)

# ─────────────────────────────────────────────
# Create output folders
# ─────────────────────────────────────────────
os.makedirs("tiles/images", exist_ok=True)
os.makedirs("tiles/masks",  exist_ok=True)

# ─────────────────────────────────────────────
# Load image + mask
# ─────────────────────────────────────────────
with rasterio.open(IMAGE_PATH) as src:
    image = src.read().astype(np.float32)
    n_bands = src.count
    print(f"Image: {src.width}×{src.height} | {n_bands} bands")
    print(f"  Raw value range: {image.min():.1f} – {image.max():.1f}")

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    print(f"Mask : {src.width}×{src.height}")
    total = mask.size
    names = ["road", "vegetation", "water", "building"]
    for c, name in enumerate(names):
        pct = (mask == c).sum() / total * 100
        print(f"  Class {c} ({name:<12}): {pct:.1f}%")

assert image.shape[1] == mask.shape[0] and image.shape[2] == mask.shape[1], \
    f"Image/mask size mismatch: image={image.shape[1:]} mask={mask.shape}"

# ─────────────────────────────────────────────
# Normalize image bands using percentile clipping
# Handles Sentinel-2 16-bit values (0–10000+) correctly
# /255.0 is WRONG for Sentinel-2 — do not use it
# ─────────────────────────────────────────────
print("\nNormalizing bands (percentile clip p2–p98)...")
norm = np.zeros_like(image, dtype=np.float32)
p2_list, p98_list = [], []

for b in range(n_bands):
    band = image[b]
    # Only use valid (non-zero) pixels for percentile calculation
    valid = band[band > 0]
    if len(valid) == 0:
        p2, p98 = 0.0, 1.0
    else:
        p2  = float(np.percentile(valid, 2))
        p98 = float(np.percentile(valid, 98))
    norm[b] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    p2_list.append(p2)
    p98_list.append(p98)

# Save stats so predict.py can apply the same normalization
np.save("band_stats.npy", {"p2": p2_list, "p98": p98_list})
print(f"  Saved band_stats.npy (needed by predict.py)")

# ─────────────────────────────────────────────
# Tile the image and mask
# ─────────────────────────────────────────────
H, W = mask.shape
tile_id = kept = skipped = 0

for y in range(0, H - TILE_SIZE + 1, STRIDE):
    for x in range(0, W - TILE_SIZE + 1, STRIDE):

        img_tile  = norm[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
        mask_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

        # Skip if wrong size (edge case)
        if img_tile.shape[1] != TILE_SIZE or img_tile.shape[2] != TILE_SIZE:
            continue

        # Skip all-nodata tiles
        if img_tile.max() == 0:
            skipped += 1
            continue

        # Skip tiles with fewer than MIN_NONROAD_PX non-road pixels
        # (keeps tiles that have at least some labelled diversity)
        if (mask_tile > 0).sum() < MIN_NONROAD_PX:
            skipped += 1
            continue

        np.save(f"tiles/images/img_{tile_id}.npy",  img_tile)
        np.save(f"tiles/masks/mask_{tile_id}.npy",  mask_tile)
        tile_id += 1
        kept += 1

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print(f"\nTiles created : {kept}")
print(f"Tiles skipped : {skipped}")

if kept == 0:
    print("\nERROR: No tiles created!")
    print("  → Check that training_mask_osm.tif exists and is not all zeros")
    print("  → Try lowering MIN_NONROAD_PX to 1")
elif kept < 50:
    print(f"\nWARNING: Only {kept} tiles — consider lowering STRIDE to 32 for more data")
else:
    print(f"\nOK — ready to run train_unet.py")

# Per-class tile coverage
print("\nClass coverage across kept tiles:")
counts = np.zeros(4, dtype=np.int64)
for i in range(kept):
    m = np.load(f"tiles/masks/mask_{i}.npy")
    for c in range(4):
        counts[c] += (m == c).sum()
total_px = counts.sum()
for c, name in enumerate(["road","vegetation","water","building"]):
    print(f"  Class {c} ({name:<12}): {counts[c]/total_px*100:.1f}%")