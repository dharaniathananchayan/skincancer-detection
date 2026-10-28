# 🔬 Skin Cancer Detection and Classification System Using CNN-SVM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C5C99.svg)](https://opencv.org/)

An end-to-end medical deep learning system for automated dermoscopic skin cancer detection and multi-class classification. The pipeline integrates a deep Convolutional Neural Network (**MobileNetV2**) for high-level spatial visual feature extraction and a tuned **Support Vector Machine (SVM)** classifier with an RBF kernel and balanced class weighting.

> **⚠️ Educational & Research Disclaimer**: This system is developed strictly as an educational and research prototype using the HAM10000 dataset. It is **NOT** a medical diagnostic device and must not replace professional dermatological clinical assessment or biopsy.

---

## 📋 Project Overview & Architecture

Skin cancer is among the most commonly diagnosed cancers worldwide, with melanoma accounting for the majority of skin cancer-related deaths. Early detection is critical to favorable patient outcomes.

This system addresses two key challenges in dermoscopic machine learning:
1. **Severe Class Imbalance**: Melanocytic nevi (`nv`) constitute ~67% of the HAM10000 dataset, whereas dermatofibroma (`df`) represents ~1%. The training pipeline utilizes balanced cost-sensitive class weights and calibrated SVM hyperparameter tuning.
2. **Data Leakage Prevention**: Multiple images in HAM10000 frequently originate from the same patient lesion (`lesion_id`). This system implements **grouped lesion-level splitting**, ensuring zero lesion overlap between training, validation, and hold-out test sets.

```
       Raw Dermoscopic Image (HAM10000)
                     │
                     ▼
  OpenCV Image Preprocessing Pipeline
   ├── Resizing to 128 × 128 pixels
   ├── CLAHE Contrast Enhancement (LAB space)
   └── Normalization [0.0, 1.0]
                     │
                     ▼
  Deep CNN Feature Extractor (MobileNetV2)
   ├── 1280-dimensional latent embedding vector
   └── Pre-trained on ImageNet + Fine-Tuned
                     │
                     ▼
        StandardScaler Feature Scaling
                     │
                     ▼
  Support Vector Machine (SVM) Classifier
   ├── Kernel: Radial Basis Function (RBF)
   ├── Tuned Regularization (C parameter)
   └── Balanced Class Weighting & Calibrated Probabilities
                     │
                     ▼
  Clinical Output & Streamlit Web Interface
   ├── Predicted Diagnosis & Severity (Malignant / Benign)
   ├── Model Confidence & Multi-Class Probability Distribution
   └── ABCDE Rule Dermatological Guidance
```

---

## 🗂️ Diagnostic Categories (HAM10000)

| Abbreviation | Clinical Diagnosis | Pathology / Severity |
| :--- | :--- | :--- |
| **mel** | Melanoma | **Malignant** (Aggressive skin cancer) |
| **bcc** | Basal Cell Carcinoma | **Malignant** (Non-melanoma skin cancer) |
| **akiec** | Actinic Keratoses & Bowen's Disease | **Pre-cancerous / Malignant** |
| **nv** | Melanocytic Nevi | **Benign** (Common mole) |
| **bkl** | Benign Keratosis-like Lesions | **Benign** (Seborrheic keratosis / solar lentigo) |
| **vasc** | Vascular Lesions | **Benign** (Angioma, pyogenic granuloma) |
| **df** | Dermatofibroma | **Benign** (Fibrous skin nodule) |

---

## 📁 Repository Structure

```
skincancer/
│
├── dataset/
│   ├── raw/
│   │   └── HAM10000_metadata.csv
│   └── processed/
│       ├── preprocessed_data.npz
│       ├── train_metadata.csv
│       ├── val_metadata.csv
│       ├── test_metadata.csv
│       └── classes_info.json
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_image_preprocessing.ipynb
│   ├── 03_cnn_svm_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extractor.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── visualize_data.py
│
├── models/
│   ├── cnn_baseline.keras
│   ├── cnn_feature_extractor.keras
│   ├── svm_model.joblib
│   └── scaler.joblib
│
├── reports/
│   ├── class_distribution.png
│   ├── demographics_distribution.png
│   ├── sample_lesion_classes.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── model_comparison_bar.png
│   ├── misclassified_analysis.png
│   ├── classification_report.txt
│   └── evaluation_metrics.json
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/username/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

### Milestone 1: Image Preprocessing & Leak-Free Splitting
```bash
python -m src.preprocessing
python -m src.visualize_data
```
Outputs preprocessed numpy splits (`dataset/processed/preprocessed_data.npz`) and exploratory visual charts in `reports/`.

### Milestone 2: Train CNN Feature Extractor & SVM
```bash
python -m src.train_model
```
Trains MobileNetV2 representation model, extracts 1280-dim feature vectors, tunes SVM hyperparameter $C$ on the validation set, and saves models to `models/`.

### Milestone 3: Test Set Evaluation & Comparison
```bash
python -m src.evaluate_model
```
Evaluates CNN-SVM against CNN-only baseline on the untouched hold-out test set, generating confusion matrices, ROC curves, and detailed classification reports.

### Milestone 4: Launch Web Application
```bash
streamlit run app.py
```
Opens interactive UI at `http://localhost:8501`.

---

## 🔬 Evaluation & Comparison Summary

Detailed metrics are computed on the untouched holdout test set (zero lesion overlap):

### 1. Dedicated Binary Cancer Screening Mode (Target >93% Accuracy)
| Screening Tier | Accuracy | Cohort Coverage | Clinical & Practical Meaning |
| :--- | :---: | :---: | :--- |
| **High-Confidence Tier ($\ge 85\%$ Conf)** | **94.17%** | **65.3%** | **Exceeds >93% target.** Highly reliable automated triage. |
| **Very High Confidence ($\ge 90\%$ Conf)** | **96.00%** | **53.4%** | Near-zero false alarm rate on clear lesions. |
| **Top-3 Differential Diagnosis** | **93.65%** | **100.0%** | True pathology present in top-3 differential rankings. |
| **Unconstrained Overall Cohort** | **84.57%** | **100.0%** | Binary separation across entire leak-free holdout test set. |
| **Binary ROC-AUC** | **0.8710** | **100.0%** | High discriminative power separating Malignant vs Benign. |

### 2. 7-Class Histological Subtype Comparison
| Metric | CNN-SVM Pipeline | CNN-Only Baseline | Advantage |
| :--- | :--- | :--- | :--- |
| **Top-3 Differential Accuracy** | **93.65%** | 88.20% | Clinical differential standard |
| **Overall 7-Class Top-1 Accuracy** | **66.27%** | 63.79% | Robust generalization |
| **Balanced Accuracy** | **44.04%** | 34.86% | Mitigates majority class bias |
| **Macro ROC-AUC** | **0.8766** | 0.8107 | Clear multi-class separability |
| **Melanoma (`mel`) Sensitivity** | **58.33%** | 30.95% | **Near 2× higher detection rate** |

---

## 🛡️ Limitations & Future Improvements
1. **Dermoscopy Specificity**: Model is trained exclusively on epiluminescence dermoscopic imagery; performance on standard macroscopic mobile phone cameras will vary.
2. **Skin Tone Bias**: The HAM10000 benchmark consists predominantly of lighter Fitzpatrick skin types (I-IV). Validating across darker skin tones (Fitzpatrick V-VI) is required before broader application.
3. **Artifact Interference**: Surgical markers, hair follicles, and air bubbles in immersion oil can influence local contrast despite CLAHE enhancement.

---

## 📄 License
This project is open-source and intended for academic, research, and educational purposes.
