"""
osm_mask.py  — Generate 4-class training mask
Classes:  0=Road | 1=Vegetation | 2=Water | 3=Building

Strategy:
  1. Compute spectral indices (NDVI, NDWI, NDBI) to separate
     vegetation / water / built-up from each other spectrally.
  2. Layer OSM features on top with correct priorities.
  3. Every pixel is assigned — no "background" class.

Priority (highest wins):
  OSM Water  > OSM Building  > OSM Road
      ↑ these override spectral when available
  Spectral indices fill in where OSM has no data.
"""

import rasterio
import numpy as np
import osmnx as ox
from rasterio.features import rasterize
from pyproj import Transformer
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CLASS_NAMES  = ["Road", "Vegetation", "Water", "Building"]
CLASS_COLORS = ["#DC5050", "#3CB44B", "#4285F4", "#FFDC50"]
N_CLASSES    = 4

# ── Sentinel-2 band indices (0-based) ──────────
# Common 4-band order:  B2=0(Blue), B3=1(Green), B4=2(Red), B8=3(NIR)
# Common 12-band order: B2=0, B3=1, B4=2, B5=3, B6=4, B7=5, B8=6, B8A=7, B11=8, B12=9...
# Adjust these if your image has a different band order!
BLUE_BAND  = 0   # B2  — used for NDWI
GREEN_BAND = 1   # B3  — used for NDWI
RED_BAND   = 2   # B4  — used for NDVI
NIR_BAND   = 3   # B8  — used for NDVI, NDBI
SWIR_BAND  = None  # B11 — used for NDBI; set to band index if available, else None

# ── Spectral thresholds ─────────────────────────
# NDVI > NDVI_VEG_THRESH  → vegetation
# Raise if urban areas are wrongly labeled green (e.g. try 0.25–0.35)
NDVI_VEG_THRESH  = 0.25

# NDWI > NDWI_WATER_THRESH → water
NDWI_WATER_THRESH = 0.1

# NDBI > NDBI_BUILT_THRESH → built-up surface (only used if SWIR available)
NDBI_BUILT_THRESH = 0.0

# ─────────────────────────────────────────────
# Load satellite image
# ─────────────────────────────────────────────
img       = rasterio.open("Sentinel_AI_Project.tif")
bounds    = img.bounds
crs       = img.crs
transform = img.transform
H, W      = img.height, img.width

print(f"✅ Image: {W}×{H} | {img.count} bands | CRS: {crs}")
print(f"   Dtype: {img.dtypes[0]}")

# Validate band indices
max_band = img.count - 1
for name, idx in [("BLUE",BLUE_BAND),("GREEN",GREEN_BAND),
                  ("RED",RED_BAND),("NIR",NIR_BAND)]:
    if idx > max_band:
        raise ValueError(f"{name}_BAND={idx} but image only has {img.count} bands. "
                         f"Adjust the band index at the top of this file.")

# ─────────────────────────────────────────────
# Read spectral bands (float32, raw DN)
# ─────────────────────────────────────────────
blue  = img.read(BLUE_BAND  + 1).astype(np.float32)
green = img.read(GREEN_BAND + 1).astype(np.float32)
red   = img.read(RED_BAND   + 1).astype(np.float32)
nir   = img.read(NIR_BAND   + 1).astype(np.float32)
swir  = img.read(SWIR_BAND  + 1).astype(np.float32) if SWIR_BAND is not None else None

# ─────────────────────────────────────────────
# Compute spectral indices
# ─────────────────────────────────────────────
eps = 1e-6

# NDVI: vegetation (high = green plants)
ndvi = (nir - red) / (nir + red + eps)

# NDWI: water (McFeeters; high = open water)
ndwi = (green - nir) / (green + nir + eps)

# NDBI: built-up index (high = concrete/rooftops); needs SWIR
ndbi = (swir - nir) / (swir + nir + eps) if swir is not None else None

print(f"\nSpectral index ranges:")
print(f"  NDVI : {ndvi.min():.2f} – {ndvi.max():.2f}  (veg threshold: >{NDVI_VEG_THRESH})")
print(f"  NDWI : {ndwi.min():.2f} – {ndwi.max():.2f}  (water threshold: >{NDWI_WATER_THRESH})")
if ndbi is not None:
    print(f"  NDBI : {ndbi.min():.2f} – {ndbi.max():.2f}  (built threshold: >{NDBI_BUILT_THRESH})")

# ─────────────────────────────────────────────
# Spectral classification masks
# ─────────────────────────────────────────────
# Strict vegetation: high NDVI AND not water
is_spectral_veg   = (ndvi > NDVI_VEG_THRESH) & (ndwi < NDWI_WATER_THRESH)

# Water: positive NDWI and NOT high NDVI
is_spectral_water = (ndwi > NDWI_WATER_THRESH) & (ndvi < 0.1)

# Built-up: only if SWIR available; else rely on OSM buildings
is_spectral_built = (ndbi > NDBI_BUILT_THRESH) if ndbi is not None else np.zeros((H,W), dtype=bool)

print(f"\nSpectral pixel counts:")
print(f"  Vegetation : {is_spectral_veg.sum():>10,}  ({is_spectral_veg.mean()*100:.1f}%)")
print(f"  Water      : {is_spectral_water.sum():>10,}  ({is_spectral_water.mean()*100:.1f}%)")
print(f"  Built-up   : {is_spectral_built.sum():>10,}  ({is_spectral_built.mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# OSM download
# ─────────────────────────────────────────────
transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
west, south = transformer.transform(bounds.left,  bounds.bottom)
east, north = transformer.transform(bounds.right, bounds.top)
cx = (east + west) / 2
cy = (north + south) / 2
dist = 3000
print(f"\n⬇️  Downloading OSM (center {cy:.4f}°N, {cx:.4f}°E, radius {dist}m)...")

def fetch_osm(tags, dist, name):
    try:
        gdf = ox.features_from_point((cy, cx), tags=tags, dist=dist)
        print(f"   {name}: {len(gdf)}")
        return gdf
    except Exception as e:
        print(f"   {name}: 0  (⚠ {e})")
        return gpd.GeoDataFrame(geometry=[])

buildings = fetch_osm({'building': True}, dist, "Buildings")
roads_gdf = None
try:
    g = ox.graph_from_point((cy, cx), dist=dist, network_type='all')
    roads_gdf = ox.graph_to_gdfs(g, nodes=False)
    print(f"   Roads: {len(roads_gdf)}")
except Exception as e:
    print(f"   Roads: 0  (⚠ {e})")
    roads_gdf = gpd.GeoDataFrame(geometry=[])

water_osm = fetch_osm({'natural': 'water'}, dist, "Water (natural)")
try:
    wr = ox.features_from_point((cy, cx), tags={'landuse': 'reservoir'}, dist=dist)
    water_osm = gpd.pd.concat([water_osm, wr]) if len(water_osm) else wr
    print(f"   Water (reservoir): added {len(wr)}")
except Exception:
    pass

# ─────────────────────────────────────────────
# Reproject OSM to image CRS
# ─────────────────────────────────────────────
def safe_reproject(gdf):
    if len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(crs)

buildings = safe_reproject(buildings)
roads_gdf = safe_reproject(roads_gdf)
water_osm = safe_reproject(water_osm)

# ─────────────────────────────────────────────
# Rasterize OSM layers
# ─────────────────────────────────────────────
def rasterize_gdf(gdf, buffer_m=0):
    if len(gdf) == 0:
        return np.zeros((H, W), dtype=bool)
    geoms = gdf.geometry.copy()
    if buffer_m > 0:
        geoms = geoms.buffer(buffer_m)
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((H, W), dtype=bool)
    return rasterize(shapes, out_shape=(H, W), transform=transform,
                     dtype=np.uint8, all_touched=True).astype(bool)

print("\n🗺  Rasterizing OSM layers...")
osm_road     = rasterize_gdf(roads_gdf, buffer_m=3)   # 3m half-width
osm_water    = rasterize_gdf(water_osm, buffer_m=0)
osm_building = rasterize_gdf(buildings, buffer_m=0)

print(f"   OSM road     pixels: {osm_road.sum():>10,}")
print(f"   OSM water    pixels: {osm_water.sum():>10,}")
print(f"   OSM building pixels: {osm_building.sum():>10,}")

# ─────────────────────────────────────────────
# Assemble 4-class mask
#
# Layer order (LAST applied = HIGHEST priority):
#
#  Base layer  → spectral built-up or Road as default
#  Layer 1     → Spectral Vegetation (NDVI)
#  Layer 2     → OSM Roads  (overrides veg where road exists)
#  Layer 3     → Spectral Water  (NDWI)
#  Layer 4     → OSM Water  (overrides everything)
#  Layer 5     → Spectral Built-up NDBI (if SWIR available)
#  Layer 6     → OSM Buildings (highest non-water priority)
#
# Net result: OSM data is trusted over spectral where available;
#   spectral fills in all the gaps (informal buildings, missing roads).
# ─────────────────────────────────────────────
print("\n🏗  Building final 4-class mask...")

mask = np.zeros((H, W), dtype=np.uint8)   # default = 0 (Road)

# Layer 1 — spectral vegetation
mask[is_spectral_veg] = 1

# Layer 2 — OSM roads override vegetation (a road is a road)
mask[osm_road] = 0

# Layer 3 — spectral water
mask[is_spectral_water] = 2

# Layer 4 — OSM water (most reliable water source)
mask[osm_water] = 2

# Layer 5 — spectral built-up (NDBI, only if SWIR present)
# Don't override water, but can override road/veg
if ndbi is not None:
    mask[is_spectral_built & ~osm_water & ~is_spectral_water] = 3

# Layer 6 — OSM buildings (highest priority after water)
mask[osm_building & ~osm_water & ~is_spectral_water] = 3

# ─────────────────────────────────────────────
# Stats & validation
# ─────────────────────────────────────────────
total = mask.size
print(f"\n📊 Final class distribution:")
for c, name in enumerate(CLASS_NAMES):
    n = (mask == c).sum()
    print(f"   Class {c} ({name:<12}): {n:>10,} px  ({n/total*100:5.1f}%)")

issues = []
for c, name in enumerate(CLASS_NAMES):
    pct = (mask == c).sum() / total * 100
    if pct == 0:
        issues.append(f"⚠  Class {c} ({name}) = 0%!")
    elif c in (2, 3) and pct < 0.5:
        issues.append(f"⚠  Class {c} ({name}) very low ({pct:.1f}%) — model may ignore it")

if issues:
    print("\n" + "\n".join(issues))
    print("\nSuggested fixes:")
    print("  → Vegetation too low: lower NDVI_VEG_THRESH (try 0.15)")
    print("  → Water too low     : lower NDWI_WATER_THRESH (try 0.05)")
    print("  → Building too low  : increase OSM dist= or add SWIR_BAND")
else:
    print("\n✅ All classes present — mask looks good!")

# ─────────────────────────────────────────────
# Save mask
# ─────────────────────────────────────────────
with rasterio.open(
    "training_mask_osm.tif", "w",
    driver="GTiff", height=H, width=W, count=1,
    dtype=mask.dtype, crs=crs, transform=transform,
) as dst:
    dst.write(mask, 1)

print("\n🎉 Saved → training_mask_osm.tif")
print("   0=Road | 1=Vegetation | 2=Water | 3=Building")

# ─────────────────────────────────────────────
# Visualisation — 4-panel check
# ─────────────────────────────────────────────
cmap4 = ListedColormap(CLASS_COLORS)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1 — final mask
im = axes[0,0].imshow(mask, cmap=cmap4, vmin=0, vmax=3, interpolation="nearest")
axes[0,0].set_title("Final 4-Class Mask")
cb = fig.colorbar(im, ax=axes[0,0], ticks=[0,1,2,3])
cb.set_ticklabels(CLASS_NAMES)

# Panel 2 — NDVI
axes[0,1].imshow(ndvi, cmap='RdYlGn', vmin=-0.3, vmax=0.6)
axes[0,1].set_title(f"NDVI  (veg = >{NDVI_VEG_THRESH})")
axes[0,1].contour(is_spectral_veg, levels=[0.5], colors='lime', linewidths=0.5)

# Panel 3 — NDWI
axes[1,0].imshow(ndwi, cmap='RdBu', vmin=-0.4, vmax=0.4)
axes[1,0].set_title(f"NDWI  (water = >{NDWI_WATER_THRESH})")
axes[1,0].contour(is_spectral_water, levels=[0.5], colors='blue', linewidths=0.5)

# Panel 4 — NDBI or class frequency bar
if ndbi is not None:
    axes[1,1].imshow(ndbi, cmap='hot', vmin=-0.3, vmax=0.4)
    axes[1,1].set_title(f"NDBI  (built = >{NDBI_BUILT_THRESH})")
else:
    counts = [(mask == c).sum() / total * 100 for c in range(N_CLASSES)]
    axes[1,1].bar(CLASS_NAMES, counts, color=CLASS_COLORS)
    axes[1,1].set_ylabel("% of pixels")
    axes[1,1].set_title("Class Distribution")
    for i, v in enumerate(counts):
        axes[1,1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("mask_preview.png", dpi=120, bbox_inches="tight")
plt.show()
print("✅ Saved mask_preview.png — check all 4 panels before running create_tiles.py")
print("\nIf vegetation/water/building panels look wrong, adjust the thresholds at the top of this file:")
print("  NDVI_VEG_THRESH, NDWI_WATER_THRESH, RED_BAND, NIR_BAND, GREEN_BAND")
 
plt.imshow(mask)
plt.show()
