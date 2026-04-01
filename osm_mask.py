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

BLUE_BAND  = 0
GREEN_BAND = 1
RED_BAND   = 2
NIR_BAND   = 3
SWIR_BAND  = None

NDVI_VEG_THRESH   = 0.25
NDWI_WATER_THRESH = 0.1
NDBI_BUILT_THRESH = 0.0

# ─────────────────────────────────────────────
# Load satellite image
# ─────────────────────────────────────────────
img       = rasterio.open("Sentinel_AI_Project.tif")
bounds    = img.bounds
crs       = img.crs
transform = img.transform
H, W      = img.height, img.width

print(f"Image: {W}x{H} | {img.count} bands | CRS: {crs}")

blue  = img.read(BLUE_BAND  + 1).astype(np.float32)
green = img.read(GREEN_BAND + 1).astype(np.float32)
red   = img.read(RED_BAND   + 1).astype(np.float32)
nir   = img.read(NIR_BAND   + 1).astype(np.float32)
swir  = img.read(SWIR_BAND  + 1).astype(np.float32) if SWIR_BAND is not None else None

eps  = 1e-6
ndvi = (nir - red)   / (nir   + red   + eps)
ndwi = (green - nir) / (green + nir   + eps)
ndbi = (swir - nir)  / (swir  + nir   + eps) if swir is not None else None

is_spectral_veg   = (ndvi > NDVI_VEG_THRESH)  & (ndwi < NDWI_WATER_THRESH)
is_spectral_water = (ndwi > NDWI_WATER_THRESH) & (ndvi < 0.1)
is_spectral_built = (ndbi > NDBI_BUILT_THRESH)  if ndbi is not None else np.zeros((H,W), bool)

transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
west, south = transformer.transform(bounds.left,  bounds.bottom)
east, north = transformer.transform(bounds.right, bounds.top)
cx = (east + west) / 2
cy = (north + south) / 2
dist = 3000

print(f"Downloading OSM data...")

def fetch_osm(tags, dist, name):
    try:
        gdf = ox.features_from_point((cy, cx), tags=tags, dist=dist)
        print(f"  {name}: {len(gdf)}")
        return gdf
    except Exception as e:
        print(f"  {name}: 0  ({e})")
        return gpd.GeoDataFrame(geometry=[])

buildings = fetch_osm({'building': True}, dist, "Buildings")
try:
    g = ox.graph_from_point((cy, cx), dist=dist, network_type='all')
    roads_gdf = ox.graph_to_gdfs(g, nodes=False)
    print(f"  Roads: {len(roads_gdf)}")
except Exception as e:
    roads_gdf = gpd.GeoDataFrame(geometry=[])

water_osm = fetch_osm({'natural': 'water'}, dist, "Water")

def safe_reproject(gdf):
    if len(gdf) == 0: return gdf
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(crs)

buildings = safe_reproject(buildings)
roads_gdf = safe_reproject(roads_gdf)
water_osm = safe_reproject(water_osm)

def rasterize_gdf(gdf, buffer_m=0):
    if len(gdf) == 0: return np.zeros((H, W), bool)
    geoms = gdf.geometry.copy()
    if buffer_m > 0: geoms = geoms.buffer(buffer_m)
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes: return np.zeros((H, W), bool)
    return rasterize(shapes, out_shape=(H, W), transform=transform,
                     dtype=np.uint8, all_touched=True).astype(bool)

osm_road     = rasterize_gdf(roads_gdf, buffer_m=3)
osm_water    = rasterize_gdf(water_osm)
osm_building = rasterize_gdf(buildings)

mask = np.zeros((H, W), dtype=np.uint8)
mask[is_spectral_veg]  = 1
mask[osm_road]         = 0
mask[is_spectral_water]= 2
mask[osm_water]        = 2
if ndbi is not None:
    mask[is_spectral_built & ~osm_water & ~is_spectral_water] = 3
mask[osm_building & ~osm_water & ~is_spectral_water] = 3

total = mask.size
print("\nFinal class distribution:")
for c, name in enumerate(CLASS_NAMES):
    n = (mask == c).sum()
    print(f"  Class {c} ({name:<12}): {n/total*100:.1f}%")

with rasterio.open("training_mask_osm.tif", "w",
    driver="GTiff", height=H, width=W, count=1,
    dtype=mask.dtype, crs=crs, transform=transform) as dst:
    dst.write(mask, 1)

print("\nSaved training_mask_osm.tif")
print("0=Road | 1=Vegetation | 2=Water | 3=Building")