# 🦟 Malaria Cell Detection & Classification
### Deep Learning Pipeline — CNN → 7-Class Multiclass Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Project Overview

This project builds a complete end-to-end deep learning pipeline to **automatically detect and classify malaria-infected blood cells** from microscopic images.

Malaria kills over **600,000 people every year**, mostly in developing countries. Manual diagnosis under a microscope is slow, expensive, and error-prone. This project uses deep learning to automate and improve that process.

The project is structured in **2 phases**:

| Phase | Task | Model Used |
|-------|------|-----------|
| Phase 1 | Binary Classification (Infected vs Uninfected) | Custom CNN |
| Phase 2 | 7-Class Multiclass Classification | ResNet50 + PCA + UMAP + SVM |

---

## Project Structure

```
malaria-detection/
│
├── phase1_cnn.ipynb              # Phase 1 — CNN binary classifier
├── phase2_multiclass.ipynb       # Phase 2 — 7-class multiclass pipeline
│
├── README.md                     # This file
│
└── outputs/
    ├── cnn_best.h5               # Saved best CNN model
    ├── sample_crops.png          # Sample cell crops per class
    ├── umap_projection.png       # UMAP 2D cluster visualisation
    ├── pca_variance.png          # PCA explained variance graph
    ├── cm_svm.png                # SVM confusion matrix
    └── class_distribution.png   # Class distribution bar chart
```

---

## 📦 Datasets Used

### Dataset 1 — NIH Malaria Cell Images
- **Source:** [Kaggle — iarunava/cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Total Images:** 27,558
- **Classes:** 2 (Parasitized, Uninfected)
- **Used for:** Phase 1 (CNN binary classification)
- **Split:** 80% Training / 20% Validation

### Dataset 2 — Malaria Bounding Boxes (BBBC041)
- **Source:** [Kaggle — kmader/malaria-bounding-boxes](https://www.kaggle.com/datasets/kmader/malaria-bounding-boxes)
- **Total Cells:** ~80,000 cropped individual cells
- **Classes:** 7 (annotated via JSON bounding boxes)
- **Used for:** Phase 2 (Multiclass pipeline)

#### 6 Valid Classes Used in Phase 2:

| Class | Type | Description |
|-------|------|-------------|
| `ring` | Infected | Earliest parasite stage — thin ring shape |
| `trophozoite` | Infected | Active feeding stage — irregular shape |
| `schizont` | Infected | About to burst — multiple nuclei visible |
| `gametocyte` | Infected | Sexual stage — banana/crescent shape |
| `red blood cell` | Uninfected | Normal healthy RBC — round and smooth |
| `leukocyte` | Uninfected | White blood cell — large with nucleus |
| `difficult` | ❌ Excluded | Ambiguous cells — excluded from training |

---

## 🧠 Model Architecture

### Phase 1 — Custom CNN (Binary Classification)
```
Input (128x128x3)
    → Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
    → Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
    → Conv2D(128, 3x3) + ReLU + MaxPooling(2x2)
    → Flatten
    → Dense(128) + ReLU
    → Dropout(0.5)
    → Dense(1) + Sigmoid        ← Binary output (0 or 1)
```
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** Up to 20 (with EarlyStopping)

### Phase 2 — Multiclass Pipeline
```
Bounding Box JSON Annotations
    → Parse JSON + Crop cells (64x64 pixels)
    → ResNet50 Feature Extraction  →  2048-dim vector per cell
    → StandardScaler normalisation
    → PCA  →  2048 dims to 128 dims  (~95% variance retained)
    → UMAP →  128 dims to 2 dims    (for visualisation only)
    → SVM Classifier (RBF kernel, class_weight=balanced)
    → 6-Class Prediction
```

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| **Python 3.10** | Core programming language |
| **TensorFlow / Keras** | Build and train the CNN model |
| **ResNet50** | Feature extractor for multiclass pipeline |
| **Scikit-learn** | PCA, SVM, metrics, preprocessing |
| **UMAP (umap-learn)** | Dimensionality reduction & 2D visualisation |
| **NumPy** | Array and matrix operations |
| **Matplotlib / Seaborn** | Plots, training curves, confusion matrices |
| **PIL (Pillow)** | Image loading, cropping and resizing |
| **JSON / Pathlib** | Parsing bounding box annotations |
| **Kaggle Notebooks** | Cloud GPU (P100) training environment |

---

## ▶️ How to Run

### Step 1 — Add Datasets on Kaggle
Add both datasets to your Kaggle notebook via **+ Add Input**:
- `iarunava/cell-images-for-detecting-malaria`
- `kmader/malaria-bounding-boxes`

### Step 2 — Install Dependencies
```python
!pip install umap-learn --quiet
```

### Step 3 — Run Notebooks in Order
```
1. phase1_cnn.ipynb          ← Binary CNN
2. phase2_multiclass.ipynb   ← 7-class pipeline
```

### Step 4 — Verify Dataset Paths
```python
import os
for dirname, dirs, files in os.walk('/kaggle/input'):
    print(dirname)
    break
```

---

## 📊 Results

| Model | Task | Accuracy |
|-------|------|----------|
| Custom CNN | Binary — 2 classes | ~85% |
| SVM on PCA Features | Multiclass — 6 classes | ~88%+ |

### Evaluation Metrics Used:
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix (raw counts + normalised %)
- UMAP cluster visualisation
- PCA explained variance curve
- Per-class accuracy bar chart

---

## 🔍 How the Model Identifies Malaria

Just like a doctor examines a blood slide under a microscope to identify which stage of malaria a patient has — this model does the same automatically:

```
Blood smear image
        ↓
CNN learns visual patterns:
  - Dark purple dots  →  parasite nucleus
  - Irregular shape   →  infected cell
  - Smooth and round  →  healthy cell
        ↓
Binary prediction: Parasitized or Uninfected
        ↓
ResNet50 extracts 2048 features per cropped cell
        ↓
PCA reduces dimensions → removes noise
        ↓
SVM classifies into one of 6 categories
        ↓
Output: "Ring stage — 94% confidence"
```

---

## 📈 Key Visualisations Generated

| Plot | Description |
|------|-------------|
| **Training Curves** | Accuracy and loss over epochs for CNN |
| **Confusion Matrix** | Correct vs incorrect predictions per class |
| **UMAP Plot** | 2D scatter showing class cluster separation |
| **PCA Variance Curve** | Components needed to retain 95% variance |
| **Class Distribution** | Cell count per class in bounding box dataset |
| **Per-Class Accuracy** | Bar chart showing accuracy for each of 6 classes |

---

## 🚀 Future Scope

- **Transfer Learning** — Add EfficientNetB0 or VGG16 for better binary accuracy
- **Mobile Deployment** — Convert model to TensorFlow Lite for field diagnosis
- **Multi-species Detection** — P. falciparum, P. vivax, P. ovale, P. malariae
- **Grad-CAM Heatmaps** — Show which part of the cell the model focuses on
- **Real-time Analysis** — Connect to microscope feed using OpenCV
- **Larger Datasets** — Train on WHO / NIH data for better generalisation

---

## 👩‍💻 Project Info

- **Type:** B.Tech Python Project
- **Domain:** Medical Image Analysis / Deep Learning
- **Platform:** Kaggle Notebooks (GPU P100)
- **Language:** Python 3.10
- **Framework:** TensorFlow 2.x + Scikit-learn

---

## 📚 References

- [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- [BBBC041 Malaria Bounding Boxes](https://www.kaggle.com/datasets/kmader/malaria-bounding-boxes)
- [ResNet Paper — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [UMAP Paper — McInnes et al., 2018](https://arxiv.org/abs/1802.03426)
- [WHO World Malaria Report 2023](https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report-2023)

---

> *"Just like how a doctor looks at a blood slide under a microscope and identifies which stage of malaria the patient has — this model does the same thing automatically using deep learning, in seconds, with high accuracy."*





Blood smear image taken under microscope
            ↓
Image fed into CNN / EfficientNetB0
            ↓
Model looks at pixel patterns:
  - Colour distribution (purple vs pink)
  - Shape of cell (round vs irregular)
  - Texture inside cell (smooth vs dotted)
  - Size relative to other cells
            ↓
ResNet50 extracts 2048 features per cell
            ↓
PCA reduces to 128 most important features
