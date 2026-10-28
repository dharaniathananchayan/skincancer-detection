"""
Helper script to generate Jupyter Notebooks for Milestones 2 & 3:
- notebooks/03_cnn_svm_training.ipynb
- notebooks/04_model_evaluation.ipynb
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

def build_training_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Milestone 2: CNN Feature Extraction & SVM Classification\n",
                "**System**: Skin Cancer Detection and Classification System Using CNN-SVM  \n",
                "**Objective**: Extract deep feature representations using a MobileNetV2 CNN backbone and train a tuned Support Vector Machine (SVM) classifier."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import tensorflow as tf\n",
                "from src.feature_extractor import build_cnn_classifier, load_feature_extractor\n",
                "from src.preprocessing import CLASS_NAMES, IDX_TO_NAME\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Preprocessed Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_path = 'dataset/processed/preprocessed_data.npz'\n",
                "data = np.load(data_path)\n",
                "X_train, y_train = data['X_train'], data['y_train']\n",
                "X_val, y_val = data['X_val'], data['y_val']\n",
                "X_test, y_test = data['X_test'], data['y_test']\n",
                "\n",
                "print(f'Train shape: {X_train.shape}, Labels: {y_train.shape}')\n",
                "print(f'Val shape:   {X_val.shape}, Labels: {y_val.shape}')\n",
                "print(f'Test shape:  {X_test.shape}, Labels: {y_test.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Inspect CNN Feature Extractor Architecture"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "extractor = load_feature_extractor('models/cnn_feature_extractor.keras')\n",
                "extractor.summary()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. SVM Hyperparameter Tuning Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tuning_file = 'reports/svm_tuning_results.json'\n",
                "if os.path.exists(tuning_file):\n",
                "    with open(tuning_file) as f:\n",
                "        tuning_data = json.load(f)\n",
                "    tuning_df = pd.DataFrame(tuning_data)\n",
                "    display(tuning_df)\n",
                "    \n",
                "    plt.figure(figsize=(8, 4))\n",
                "    sns.lineplot(data=tuning_df, x='C', y='val_balanced_accuracy', marker='o', lw=2, color='crimson')\n",
                "    plt.title('SVM Hyperparameter Tuning (C parameter vs Validation Balanced Accuracy)', fontweight='bold')\n",
                "    plt.xscale('log')\n",
                "    plt.xlabel('Regularization Parameter C')\n",
                "    plt.ylabel('Validation Balanced Accuracy')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Test Single Image Inference Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.predict import predict_skin_lesion\n",
                "sample_img = X_test[0]\n",
                "res = predict_skin_lesion(sample_img)\n",
                "print(f\"Predicted Diagnosis: {res['predicted_name']} ({res['predicted_code']})\")\n",
                "print(f\"Confidence:          {res['confidence']}%\")\n",
                "print(f\"Severity Category:   {res['severity']}\")\n",
                "print(f\"Risk Recommendation: {res['risk_level']}\")\n",
                "\n",
                "plt.figure(figsize=(4, 4))\n",
                "plt.imshow(res['preprocessed_image'])\n",
                "plt.title(f\"{res['predicted_name']} ({res['confidence']}%)\", fontweight='bold')\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells, "notebooks/03_cnn_svm_training.ipynb")

def build_evaluation_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Milestone 3: Model Evaluation & Performance Optimization\n",
                "**System**: Skin Cancer Detection and Classification System Using CNN-SVM  \n",
                "**Objective**: Rigorous quantitative evaluation on untouched test set: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, and Comparison against CNN-only Baseline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from PIL import Image\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Classification Metrics Summary"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metrics_path = 'reports/evaluation_metrics.json'\n",
                "with open(metrics_path) as f:\n",
                "    metrics = json.load(f)\n",
                "\n",
                "comp_table = pd.DataFrame({\n",
                "    'Metric': list(metrics['svm_model'].keys()),\n",
                "    'CNN-SVM': [round(v, 4) for v in metrics['svm_model'].values()],\n",
                "    'CNN Baseline': [round(v, 4) for v in metrics['cnn_baseline'].values()]\n",
                "})\n",
                "display(comp_table)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Classification Report"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "report_file = 'reports/classification_report.txt'\n",
                "with open(report_file) as f:\n",
                "    print(f.read())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Confusion Matrix and Error Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cm_img = Image.open('reports/confusion_matrix.png')\n",
                "plt.figure(figsize=(16, 8))\n",
                "plt.imshow(cm_img)\n",
                "plt.axis('off')\n",
                "plt.title('Confusion Matrix: Normalized & Raw Counts', fontsize=14, fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Multi-Class ROC-AUC Curves"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "roc_img = Image.open('reports/roc_curves.png')\n",
                "plt.figure(figsize=(10, 8))\n",
                "plt.imshow(roc_img)\n",
                "plt.axis('off')\n",
                "plt.title('Multi-Class ROC-AUC Curves', fontsize=14, fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Misclassified Lesion Case Studies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mis_path = 'reports/misclassified_analysis.png'\n",
                "if os.path.exists(mis_path):\n",
                "    mis_img = Image.open(mis_path)\n",
                "    plt.figure(figsize=(15, 9))\n",
                "    plt.imshow(mis_img)\n",
                "    plt.axis('off')\n",
                "    plt.title('Sample Misclassified Cases and Failure Analysis', fontsize=14, fontweight='bold')\n",
                "    plt.show()"
            ]
        }
    ]
    create_notebook(cells, "notebooks/04_model_evaluation.ipynb")

if __name__ == "__main__":
    build_training_notebook()
    build_evaluation_notebook()
