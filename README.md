# 🌍 Satellite Land Cover Classification using Deep Learning

## 📌 Overview

This project performs **pixel-wise land cover classification** on satellite imagery using **Sentinel-2 data** and a **U-Net deep learning model**.

It classifies each pixel into:

* 🛣️ Road
* 🌱 Vegetation
* 💧 Water
* 🏢 Building

The system combines **spectral indices (NDVI, NDWI, NDBI)** with **OpenStreetMap (OSM) data** to generate accurate training labels.

---

## 🚀 Key Features

* Uses real **Sentinel-2 satellite imagery**
* Automatic label generation using **OSM + spectral indices**
* Multi-class segmentation (**4 classes**)
* Tile-based training for large images
* Optimized **U-Net with MobileNetV2 encoder**
* End-to-end pipeline (data → training → prediction)

---

## 🧠 Methodology

### 1️⃣ Data Input

* Input image: Sentinel-2 `.tif`
* Multi-band satellite data (RGB + NIR)

---

### 2️⃣ Spectral Feature Extraction

* **NDVI (Normalized Difference Vegetation Index)**
  → Detects vegetation
* **NDWI (Normalized Difference Water Index)**
  → Detects water
* **NDBI (Built-up Index)** *(optional)*
  → Detects buildings

---

### 3️⃣ Automatic Mask Generation

Using :

* Combines:

  * OSM data (roads, buildings, water)
  * Spectral indices (NDVI, NDWI)
* Priority-based labeling:

  ```
  OSM Water > OSM Building > OSM Road > Spectral Data
  ```

---

### 4️⃣ Tile Generation

Using :

* Large image split into **128×128 tiles**
* Overlapping tiles (stride = 64)
* Normalization using percentile clipping

---

### 5️⃣ Model Training

Using :

* Model: **U-Net (MobileNetV2 encoder)**
* Loss:

  * CrossEntropy Loss
  * Dice Loss
* Batch size: 32
* Early stopping used

---

### 6️⃣ Prediction

Using :

* Predicts segmentation on tiles
* Generates colored output masks
* Displays class distribution

---

## 📊 Results

### 🛰️ Input Image + NDVI + Mask

![Input + NDVI + Mask](outputs/mask_preview.png)

---

### 🧩 Final Segmentation Mask + Analysis

![Mask Analysis](outputs/prediction_output.png)

---

### 🔍 Tile-wise Predictions

* Shows RGB vs predicted segmentation
* Helps visualize model performance on local regions

---

## 📈 Class Distribution Example

* Road: ~82%
* Vegetation: ~0–20%
* Water: ~1%
* Building: ~16%

---

## 📂 Project Structure

```bash
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

## 🔮 Future Improvements

* Use Transformer models (SegFormer)
* Improve small object detection (roads)
* Use multi-temporal satellite data
* Deploy as a web application

---



Give it a ⭐ on GitHub!
