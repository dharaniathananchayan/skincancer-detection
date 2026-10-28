"""
Dataset Exploration and Visualization Script for Milestone 1.
Generates exploratory figures, class distribution charts, sample image comparisons,
and writes statistical summaries to reports/.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2

from src.preprocessing import (
    index_image_paths,
    CLASS_NAMES,
    CLASS_TO_IDX,
    apply_clahe,
    preprocess_single_image
)

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_exploratory_reports(
    metadata_csv: str = "HAM10000_metadata.csv",
    reports_dir: str = "reports"
):
    os.makedirs(reports_dir, exist_ok=True)
    if not os.path.exists(metadata_csv):
        metadata_csv = os.path.join("dataset", "raw", "HAM10000_metadata.csv")

    df = pd.read_csv(metadata_csv)
    image_map = index_image_paths()

    # 1. Class Distribution Bar & Pie Plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Count per class
    plt.subplot(1, 2, 1)
    class_counts = df["dx"].value_counts()
    names = [f"{CLASS_NAMES[k]} ({k})" for k in class_counts.index]
    palette = sns.color_palette("viridis", len(class_counts))
    bars = plt.bar(names, class_counts.values, color=palette)
    plt.title("Skin Cancer Class Distribution (HAM10000)", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 50, f"{h}", ha="center", va="bottom", fontsize=9)

    # Subplot 2: Pie chart percentage
    plt.subplot(1, 2, 2)
    plt.pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%",
            startangle=140, colors=palette)
    plt.title("Class Proportion & Imbalance Analysis", fontsize=13, fontweight="bold")

    class_dist_path = os.path.join(reports_dir, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {class_dist_path}")

    # 2. Demographics (Age and Localization)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    valid_age = df.dropna(subset=["age"])
    sns.histplot(data=valid_age, x="age", hue="dx", multiple="stack", bins=16, palette="tab10")
    plt.title("Age Distribution by Diagnosis", fontsize=13, fontweight="bold")
    plt.xlabel("Age (Years)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    loc_order = df["localization"].value_counts().index
    sns.countplot(data=df, y="localization", order=loc_order, palette="mako")
    plt.title("Lesion Localization across Body Sites", fontsize=13, fontweight="bold")
    plt.xlabel("Count")
    plt.ylabel("Anatomical Site")

    demographics_path = os.path.join(reports_dir, "demographics_distribution.png")
    plt.savefig(demographics_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {demographics_path}")

    # 3. Sample Lesion Images (Original vs CLAHE Preprocessed)
    classes = list(CLASS_NAMES.keys())
    fig, axes = plt.subplots(len(classes), 2, figsize=(7, 2.5 * len(classes)))
    
    for idx, c in enumerate(classes):
        sample_row = df[df["dx"] == c].iloc[0]
        img_id = sample_row["image_id"]
        img_path = image_map.get(img_id)
        if img_path and os.path.exists(img_path):
            raw_bgr = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_resized = cv2.resize(raw_rgb, (128, 128))
            clahe_img = apply_clahe(raw_resized)

            axes[idx, 0].imshow(raw_resized)
            axes[idx, 0].set_title(f"Original: {CLASS_NAMES[c]} ({c})", fontsize=10)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(clahe_img)
            axes[idx, 1].set_title(f"CLAHE Enhanced (128x128)", fontsize=10)
            axes[idx, 1].axis("off")

    sample_path = os.path.join(reports_dir, "sample_lesion_classes.png")
    plt.savefig(sample_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {sample_path}")

    # 4. Summary JSON
    summary_stats = {
        "dataset_name": "HAM10000 (Human Against Machine with 10000 training images)",
        "total_images": len(df),
        "total_unique_lesions": int(df["lesion_id"].nunique()),
        "image_formats": "JPEG (24-bit RGB)",
        "standard_resized_resolution": [128, 128, 3],
        "class_counts": {k: int(v) for k, v in df["dx"].value_counts().items()},
        "class_full_names": CLASS_NAMES,
        "imbalance_ratio_max_to_min": float(class_counts.max() / class_counts.min()),
        "demographics": {
            "mean_age": float(df["age"].mean()),
            "median_age": float(df["age"].median()),
            "male_count": int((df["sex"] == "male").sum()),
            "female_count": int((df["sex"] == "female").sum()),
            "unknown_gender": int((df["sex"] == "unknown").sum())
        }
    }

    summary_file = os.path.join(reports_dir, "dataset_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"Saved: {summary_file}")
    return summary_stats


if __name__ == "__main__":
    generate_exploratory_reports()
