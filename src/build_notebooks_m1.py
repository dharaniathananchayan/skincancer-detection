"""
Helper script to generate Jupyter Notebooks for Milestone 1:
- notebooks/01_dataset_exploration.ipynb
- notebooks/02_image_preprocessing.ipynb
"""

import json
import os

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook: {filepath}")

def build_exploration_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Milestone 1: Skin Cancer Dataset Exploration (HAM10000)\n",
                "**System**: Skin Cancer Detection and Classification System Using CNN-SVM  \n",
                "**Objective**: Explore dataset labels, class distribution, lesion characteristics, and patient demographics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import cv2\n",
                "from src.preprocessing import CLASS_NAMES, index_image_paths\n",
                "\n",
                "# Set style\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Metadata and Index Images"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metadata_path = 'dataset/raw/HAM10000_metadata.csv'\n",
                "if not os.path.exists(metadata_path):\n",
                "    metadata_path = 'HAM10000_metadata.csv'\n",
                "\n",
                "df = pd.read_csv(metadata_path)\n",
                "print(f'Total records: {len(df)}')\n",
                "print(f'Unique lesions: {df[\"lesion_id\"].nunique()}')\n",
                "image_map = index_image_paths()\n",
                "print(f'Total available images indexed: {len(image_map)}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Analyze Diagnostic Classes & Class Imbalance\n",
                "The 7 diagnostic classes in HAM10000 are:\n",
                "- `nv`: Melanocytic Nevi (benign mole)\n",
                "- `mel`: Melanoma (malignant)\n",
                "- `bkl`: Benign Keratosis (benign)\n",
                "- `bcc`: Basal Cell Carcinoma (malignant)\n",
                "- `akiec`: Actinic Keratoses (pre-malignant / malignant)\n",
                "- `vasc`: Vascular Lesions (benign)\n",
                "- `df`: Dermatofibroma (benign)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class_counts = df['dx'].value_counts()\n",
                "class_df = pd.DataFrame({\n",
                "    'Code': class_counts.index,\n",
                "    'Condition': [CLASS_NAMES[c] for c in class_counts.index],\n",
                "    'Count': class_counts.values,\n",
                "    'Percentage': (class_counts.values / len(df) * 100).round(2)\n",
                "})\n",
                "display(class_df)\n",
                "\n",
                "plt.figure(figsize=(10, 4))\n",
                "sns.barplot(data=class_df, x='Count', y='Condition', palette='viridis')\n",
                "plt.title('HAM10000 Class Distribution', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Number of Images')\n",
                "plt.ylabel('Diagnostic Category')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Patient Demographics & Anatomical Localization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Age distribution\n",
                "sns.histplot(df['age'].dropna(), kde=True, ax=axes[0], color='teal', bins=16)\n",
                "axes[0].set_title('Patient Age Distribution', fontweight='bold')\n",
                "axes[0].set_xlabel('Age')\n",
                "\n",
                "# Localization\n",
                "loc_counts = df['localization'].value_counts()\n",
                "sns.barplot(x=loc_counts.values, y=loc_counts.index, ax=axes[1], palette='rocket')\n",
                "axes[1].set_title('Lesion Localization Across Body Sites', fontweight='bold')\n",
                "axes[1].set_xlabel('Count')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visualize Sample Images from Each Diagnostic Category"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 7, figsize=(20, 3))\n",
                "for idx, (code, name) in enumerate(CLASS_NAMES.items()):\n",
                "    sample_img_id = df[df['dx'] == code].iloc[0]['image_id']\n",
                "    img_path = image_map[sample_img_id]\n",
                "    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)\n",
                "    axes[idx].imshow(img)\n",
                "    axes[idx].set_title(f\"{code.upper()}\\n{name}\", fontsize=9)\n",
                "    axes[idx].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells, "notebooks/01_dataset_exploration.ipynb")

def build_preprocessing_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Milestone 1: Image Preprocessing, Enhancement, & Data Splitting\n",
                "**System**: Skin Cancer Detection and Classification System Using CNN-SVM  \n",
                "**Objective**: Resize, apply CLAHE contrast enhancement, normalize pixel values, perform leak-free lesion-level splitting, and augment training data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import cv2\n",
                "from src.preprocessing import (\n",
                "    apply_clahe,\n",
                "    preprocess_single_image,\n",
                "    augment_image_array,\n",
                "    create_leak_free_splits,\n",
                "    index_image_paths,\n",
                "    CLASS_NAMES\n",
                ")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Demonstrate CLAHE (Contrast Limited Adaptive Histogram Equalization)\n",
                "Dermoscopic images often have uneven lighting and low lesion-to-skin contrast.  \n",
                "CLAHE enhances local contrast in the L-channel of the LAB color space without altering colors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "image_map = index_image_paths()\n",
                "sample_id = list(image_map.keys())[0]\n",
                "sample_path = image_map[sample_id]\n",
                "\n",
                "raw_bgr = cv2.imread(sample_path)\n",
                "raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)\n",
                "resized = cv2.resize(raw_rgb, (128, 128))\n",
                "enhanced = apply_clahe(resized)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n",
                "axes[0].imshow(resized)\n",
                "axes[0].set_title('Original Resized (128x128)')\n",
                "axes[0].axis('off')\n",
                "\n",
                "axes[1].imshow(enhanced)\n",
                "axes[1].set_title('CLAHE Enhanced (128x128)')\n",
                "axes[1].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Demonstrate Data Augmentation\n",
                "Augmentations include random horizontal flip, vertical flip, and rotations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 5, figsize=(15, 3))\n",
                "axes[0].imshow(enhanced)\n",
                "axes[0].set_title('Original Base')\n",
                "axes[0].axis('off')\n",
                "\n",
                "for i in range(1, 5):\n",
                "    aug = augment_image_array(enhanced)\n",
                "    axes[i].imshow(aug)\n",
                "    axes[i].set_title(f'Augmented #{i}')\n",
                "    axes[i].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Verify Leak-Free Lesion-Level Dataset Splitting"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metadata_path = 'dataset/raw/HAM10000_metadata.csv'\n",
                "if not os.path.exists(metadata_path):\n",
                "    metadata_path = 'HAM10000_metadata.csv'\n",
                "df = pd.read_csv(metadata_path)\n",
                "train_df, val_df, test_df = create_leak_free_splits(df)\n",
                "\n",
                "print(f'Train: {len(train_df)} images ({train_df[\"lesion_id\"].nunique()} lesions)')\n",
                "print(f'Val:   {len(val_df)} images ({val_df[\"lesion_id\"].nunique()} lesions)')\n",
                "print(f'Test:  {len(test_df)} images ({test_df[\"lesion_id\"].nunique()} lesions)')\n",
                "\n",
                "# Check intersection\n",
                "train_lesions = set(train_df['lesion_id'])\n",
                "val_lesions = set(val_df['lesion_id'])\n",
                "test_lesions = set(test_df['lesion_id'])\n",
                "print('Train-Val overlap:', len(train_lesions.intersection(val_lesions)))\n",
                "print('Train-Test overlap:', len(train_lesions.intersection(test_lesions)))\n",
                "print('Val-Test overlap:', len(val_lesions.intersection(test_lesions)))\n"
            ]
        }
    ]
    create_notebook(cells, "notebooks/02_image_preprocessing.ipynb")

if __name__ == "__main__":
    build_exploration_notebook()
    build_preprocessing_notebook()
