"""
Comprehensive Model Evaluation and Comparison Module (Milestone 3).
Evaluates the trained CNN-SVM pipeline and CNN-only baseline on the untouched test set.
Computes Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Generates Confusion Matrices, ROC Curves, Comparison Visualizations,
Error Analysis of Misclassified Images, and exports reports.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

from src.preprocessing import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IDX_TO_NAME,
    SEVERITY_MAPPING
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def evaluate_models_on_test_set(
    data_path: str = os.path.join("dataset", "processed", "preprocessed_data.npz"),
    models_dir: str = "models",
    reports_dir: str = "reports"
):
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load test data
    logger.info("Loading test dataset from %s...", data_path)
    data = np.load(data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    num_classes = len(np.unique(y_test))
    class_codes = [IDX_TO_CLASS[i] for i in range(num_classes)]
    class_display_names = [f"{IDX_TO_NAME[i]} ({IDX_TO_CLASS[i]})" for i in range(num_classes)]

    # 2. Load trained models
    cnn_model_path = os.path.join(models_dir, "cnn_baseline.keras")
    extractor_path = os.path.join(models_dir, "cnn_feature_extractor.keras")
    svm_path = os.path.join(models_dir, "svm_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")

    cnn_model = tf.keras.models.load_model(cnn_model_path)
    extractor = tf.keras.models.load_model(extractor_path, compile=False)
    svm_model = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)

    # 3. Predictions: CNN Baseline
    logger.info("Generating predictions for CNN Baseline...")
    cnn_probs = cnn_model.predict(X_test, batch_size=64, verbose=0)
    cnn_preds = np.argmax(cnn_probs, axis=1)

    # 4. Predictions: CNN-SVM
    logger.info("Extracting features and generating predictions for CNN-SVM...")
    test_features = extractor.predict(X_test, batch_size=64, verbose=0)
    test_features_scaled = scaler.transform(test_features)
    svm_preds = svm_model.predict(test_features_scaled)
    svm_probs = svm_model.predict_proba(test_features_scaled)

    # 5. Compute Metrics for both models
    def get_metrics_dict(y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        
        # Multi-class ROC AUC (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=range(num_classes))
        try:
            roc_macro = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
            roc_weighted = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        except Exception as e:
            logger.warning("ROC-AUC computation notice: %s", e)
            roc_macro, roc_weighted = 0.0, 0.0

        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1_m),
            "roc_auc_macro": float(roc_macro),
            "roc_auc_weighted": float(roc_weighted)
        }

    svm_metrics = get_metrics_dict(y_test, svm_preds, svm_probs)
    cnn_metrics = get_metrics_dict(y_test, cnn_preds, cnn_probs)

    logger.info("CNN-SVM: Accuracy=%.4f, BalAcc=%.4f, Weighted-F1=%.4f, Macro-F1=%.4f",
                svm_metrics["accuracy"], svm_metrics["balanced_accuracy"],
                svm_metrics["f1_weighted"], svm_metrics["f1_macro"])
    logger.info("CNN Baseline: Accuracy=%.4f, BalAcc=%.4f, Weighted-F1=%.4f, Macro-F1=%.4f",
                cnn_metrics["accuracy"], cnn_metrics["balanced_accuracy"],
                cnn_metrics["f1_weighted"], cnn_metrics["f1_macro"])

    # 6. Save Classification Report (CNN-SVM)
    clf_report_str = classification_report(
        y_test,
        svm_preds,
        target_names=class_display_names,
        digits=4,
        zero_division=0
    )
    report_txt_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SKIN CANCER CLASSIFICATION REPORT (CNN-SVM MODEL ON TEST SET)\n")
        f.write("=" * 80 + "\n\n")
        f.write(clf_report_str)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("CNN-ONLY BASELINE CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(classification_report(y_test, cnn_preds, target_names=class_display_names, digits=4, zero_division=0))
    logger.info("Saved classification reports to %s", report_txt_path)

    # 7. Save Evaluation Metrics JSON
    metrics_export = {
        "svm_model": svm_metrics,
        "cnn_baseline": cnn_metrics,
        "test_sample_count": len(y_test),
        "class_mapping": CLASS_TO_IDX
    }
    metrics_json_path = os.path.join(reports_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_export, f, indent=4)
    logger.info("Saved evaluation metrics JSON to %s", metrics_json_path)

    # 8. Confusion Matrix Heatmaps
    cm_svm = confusion_matrix(y_test, svm_preds)
    cm_svm_norm = cm_svm.astype("float") / cm_svm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.split()[0] for c in class_codes],
                yticklabels=class_display_names, ax=axes[0])
    axes[0].set_title("CNN-SVM Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Diagnosis")
    axes[0].set_ylabel("True Diagnosis")

    sns.heatmap(cm_svm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[c.split()[0] for c in class_codes],
                yticklabels=class_display_names, ax=axes[1])
    axes[1].set_title("CNN-SVM Confusion Matrix (Normalized Recall)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Diagnosis")
    axes[1].set_ylabel("True Diagnosis")

    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix figure to %s", cm_path)

    # 9. Model Comparison Bar Chart
    comparison_df = pd.DataFrame({
        "Metric": ["Overall Accuracy", "Balanced Accuracy", "Weighted F1", "Macro F1", "Weighted Recall", "ROC-AUC (Macro)"],
        "CNN-SVM": [
            svm_metrics["accuracy"],
            svm_metrics["balanced_accuracy"],
            svm_metrics["f1_weighted"],
            svm_metrics["f1_macro"],
            svm_metrics["recall_weighted"],
            svm_metrics["roc_auc_macro"]
        ],
        "CNN Baseline": [
            cnn_metrics["accuracy"],
            cnn_metrics["balanced_accuracy"],
            cnn_metrics["f1_weighted"],
            cnn_metrics["f1_macro"],
            cnn_metrics["recall_weighted"],
            cnn_metrics["roc_auc_macro"]
        ]
    })
    comp_melted = pd.melt(comparison_df, id_vars=["Metric"], var_name="Model", value_name="Score")

    plt.figure(figsize=(10, 5))
    bar_plot = sns.barplot(data=comp_melted, x="Metric", y="Score", hue="Model", palette=["#2b5c8f", "#d95f02"])
    plt.title("Performance Comparison: CNN-SVM vs. CNN Baseline", fontsize=14, fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    for p in bar_plot.patches:
        height = p.get_height()
        if height > 0:
            bar_plot.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                              ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    plt.xticks(rotation=20, ha="right")
    comp_fig_path = os.path.join(reports_dir, "model_comparison_bar.png")
    plt.savefig(comp_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved comparison chart to %s", comp_fig_path)

    # 10. Multi-class ROC Curves
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("tab10", num_classes)
    for i in range(num_classes):
        fpr_svm, tpr_svm, _ = roc_curve(y_test_bin[:, i], svm_probs[:, i])
        roc_auc_i = auc(fpr_svm, tpr_svm)
        plt.plot(fpr_svm, tpr_svm, color=colors[i], lw=2,
                 label=f"{class_codes[i]} - {IDX_TO_NAME[i]} (AUC = {roc_auc_i:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("One-vs-Rest ROC Curves for CNN-SVM Classifier", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    roc_fig_path = os.path.join(reports_dir, "roc_curves.png")
    plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve figure to %s", roc_fig_path)

    # 11. Misclassified Image Examination & Root Cause Analysis
    misclassified_indices = np.where(svm_preds != y_test)[0]
    logger.info("Total misclassified test images: %d / %d (%.2f%%)",
                len(misclassified_indices), len(y_test), (len(misclassified_indices)/len(y_test))*100)

    if len(misclassified_indices) > 0:
        sample_mis = misclassified_indices[:min(6, len(misclassified_indices))]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for plot_idx, img_idx in enumerate(sample_mis):
            img = (X_test[img_idx] * 255.0).astype(np.uint8)
            true_label = IDX_TO_NAME[y_test[img_idx]]
            true_code = IDX_TO_CLASS[y_test[img_idx]]
            pred_label = IDX_TO_NAME[svm_preds[img_idx]]
            pred_code = IDX_TO_CLASS[svm_preds[img_idx]]
            conf = svm_probs[img_idx][svm_preds[img_idx]] * 100

            axes[plot_idx].imshow(img)
            axes[plot_idx].set_title(
                f"True: {true_label} ({true_code})\nPred: {pred_label} ({pred_code}) [{conf:.1f}%]",
                fontsize=10, color="darkred", fontweight="bold"
            )
            axes[plot_idx].axis("off")

        mis_fig_path = os.path.join(reports_dir, "misclassified_analysis.png")
        plt.savefig(mis_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved misclassified sample analysis to %s", mis_fig_path)

    logger.info("Evaluation complete! All reports, graphs, and metrics saved.")
    return {
        "svm_metrics": svm_metrics,
        "cnn_metrics": cnn_metrics,
        "num_misclassified": int(len(misclassified_indices))
    }


if __name__ == "__main__":
    evaluate_models_on_test_set()
