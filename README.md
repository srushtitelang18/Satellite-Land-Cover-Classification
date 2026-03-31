# 🌍 Satellite Land Cover Classification using U-Net

🚀 AI-based land cover classification using Sentinel-2 satellite imagery, spectral indices (NDVI, NDWI), OpenStreetMap (OSM) data, and a U-Net deep learning model.

---

## 📌 Overview

This project performs **pixel-wise land cover classification** on satellite imagery using **Sentinel-2 data** and a **U-Net deep learning model**.

It classifies each pixel into:

* 🛣️ Road
* 🌱 Vegetation
* 💧 Water
* 🏢 Building

The system combines **spectral indices (NDVI, NDWI, NDBI)** with **OpenStreetMap (OSM) data** to generate accurate training labels.

---

## 🎯 Problem Statement

Accurately classify land cover types (roads, vegetation, water, buildings) from satellite imagery using deep learning and geospatial data.

---

## 🚀 Key Features

* Uses real **Sentinel-2 satellite imagery**
* Automatic label generation using **OSM + spectral indices**
* Multi-class segmentation (**4 classes**)
* Tile-based training for large images
* Optimized **U-Net with MobileNetV2 encoder**
* End-to-end pipeline (data → training → prediction)

---

## 🔄 Workflow

Satellite Image → NDVI/NDWI → OSM Mask → Tile Generation → U-Net Training → Prediction

---

## 🧠 Methodology

### 1️⃣ Data Input

* Input image: Sentinel-2 `.tif`
* Multi-band satellite data (RGB + NIR)

---

### 2️⃣ Spectral Feature Extraction

* **NDVI (Normalized Difference Vegetation Index)** → Vegetation detection
* **NDWI (Normalized Difference Water Index)** → Water detection
* **NDBI (Built-up Index)** *(optional)* → Building detection

---

### 3️⃣ Automatic Mask Generation

Using `osm_mask.py`:

* Combines:

  * OSM data (roads, buildings, water)
  * Spectral indices (NDVI, NDWI)
* Priority-based labeling:

```
OSM Water > OSM Building > OSM Road > Spectral Data
```

---

### 4️⃣ Tile Generation

Using `create_tiles.py`:

* Large image split into **128×128 tiles**
* Overlapping tiles (stride = 64)
* Normalization using percentile clipping

---

### 5️⃣ Model Training

Using `train_unet.py`:

* Model: **U-Net (MobileNetV2 encoder)**
* Loss:

  * CrossEntropy Loss
  * Dice Loss
* Batch size: 32
* Early stopping used

---

### 6️⃣ Prediction

Using `predict.py`:

* Predicts segmentation on tiles
* Generates colored output masks
* Displays class distribution

---

## 📊 Results

### 🛰️ Input + NDVI + Mask

![Mask Preview](outputs/mask_preview.png)

---

### 🧩 Final Segmentation Output

![Prediction Output](outputs/prediction_output.png)

---

## 📈 Class Distribution Example

* Road: ~82%
* Vegetation: ~0–20%
* Water: ~1%
* Building: ~16%

---

## 📂 Project Structure

```
Satellite-LandCover-Classification/
│
├── src/
│   ├── osm_mask.py
│   ├── create_tiles.py
│   ├── train_unet.py
│   ├── predict.py
│
├── outputs/
│   ├── mask_preview.png
│   ├── prediction_output.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/srushtitelang18/Satellite-Land-Cover-Classification.git
cd Satellite-Land-Cover-Classification
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python src/osm_mask.py
python src/create_tiles.py
python src/train_unet.py
python src/predict.py
```

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Rasterio
* OSMnx
* GeoPandas
* NumPy
* Matplotlib

---

## 📚 Key Learnings

* Integration of geospatial data (OSM) with satellite imagery
* Feature extraction using NDVI & NDWI
* Semantic segmentation using U-Net
* Handling large images using tiling strategy

---

## 🔮 Future Improvements

* Use Transformer models (SegFormer)
* Improve small object detection (roads)
* Use multi-temporal satellite data
* Deploy as a web application

---

## 👩‍💻 Author

**Srushti Telang**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
