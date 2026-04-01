"""
create_tiles.py
4 classes: 0=Road | 1=Vegetation | 2=Water | 3=Building
Compatible with SegFormer / DeepLabV3+ / FPN approach in train.py
"""

import rasterio
import numpy as np
import os

IMAGE_PATH     = "Sentinel_AI_Project.tif"
MASK_PATH      = "training_mask_osm.tif"
TILE_SIZE      = 128
STRIDE         = 64
MIN_NONROAD_PX = 5

os.makedirs("tiles/images", exist_ok=True)
os.makedirs("tiles/masks",  exist_ok=True)

with rasterio.open(IMAGE_PATH) as src:
    image   = src.read().astype(np.float32)
    n_bands = src.count
    print(f"Image: {src.width}x{src.height} | {n_bands} bands")
    print(f"  Raw range: {image.min():.1f} – {image.max():.1f}")

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    print(f"Mask : {src.width}x{src.height}")
    total = mask.size
    for c, name in enumerate(["road","vegetation","water","building"]):
        print(f"  Class {c} ({name:<12}): {(mask==c).sum()/total*100:.1f}%")

# Per-band percentile normalization (correct for 16-bit Sentinel-2)
print("\nNormalizing bands...")
norm = np.zeros_like(image)
p2_list, p98_list = [], []
for b in range(n_bands):
    band  = image[b]
    valid = band[band > 0]
    p2    = float(np.percentile(valid, 2))  if len(valid) else 0.0
    p98   = float(np.percentile(valid, 98)) if len(valid) else 1.0
    norm[b] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    p2_list.append(p2)
    p98_list.append(p98)

np.save("band_stats.npy", {"p2": p2_list, "p98": p98_list})
print("  Saved band_stats.npy")

H, W = mask.shape
tile_id = kept = skipped = 0

for y in range(0, H - TILE_SIZE + 1, STRIDE):
    for x in range(0, W - TILE_SIZE + 1, STRIDE):
        img_tile  = norm[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
        mask_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

        if img_tile.shape[1] != TILE_SIZE: continue
        if img_tile.max() == 0: skipped += 1; continue
        if (mask_tile > 0).sum() < MIN_NONROAD_PX: skipped += 1; continue

        np.save(f"tiles/images/img_{tile_id}.npy", img_tile)
        np.save(f"tiles/masks/mask_{tile_id}.npy", mask_tile)
        tile_id += 1
        kept    += 1

print(f"\nTiles created : {kept}")
print(f"Tiles skipped : {skipped}")

if kept == 0:
    print("ERROR: No tiles created. Check mask file.")
else:
    print("Ready to run train.py")