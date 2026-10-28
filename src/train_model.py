"""
Model Training and Optimization Module.
Handles:
1. Representation learning with CNN (MobileNetV2) on class-imbalanced skin lesion images.
2. Saving CNN baseline model and CNN feature extractor.
3. Feature extraction across train, validation, and test sets.
4. Feature standardization (StandardScaler).
5. SVM classifier training and validation hyperparameter tuning.
6. Saving trained artifacts (cnn_feature_extractor.keras, svm_model.joblib, scaler.joblib).
"""

import os
import json
import logging
import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.feature_extractor import build_cnn_classifier, extract_features_batch, save_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(
    data_npz_path: str = os.path.join("dataset", "processed", "preprocessed_data.npz"),
    models_dir: str = "models",
    reports_dir: str = "reports",
    cnn_epochs: int = 4,
    batch_size: int = 64
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger.info("Loading preprocessed dataset from %s...", data_npz_path)
    data = np.load(data_npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    logger.info("Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)

    # 1. Compute class weights to counter severe imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info("Computed class weights: %s", class_weight_dict)

    # 2. Build CNN and Feature Extractor
    full_model, feature_extractor = build_cnn_classifier(
        input_shape=(128, 128, 3),
        num_classes=len(classes),
        learning_rate=1e-4,
        fine_tune_layers=25
    )

    # Callbacks for robust training
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    # 3. Train CNN for representation learning
    logger.info("Training CNN for %d epochs...", cnn_epochs)
    history = full_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cnn_epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(os.path.join(reports_dir, "cnn_training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=4)

    # 4. Save CNN Baseline and Feature Extractor
    save_models(full_model, feature_extractor, models_dir=models_dir)

    # 5. Extract Deep Feature Vectors
    logger.info("Extracting feature vectors from CNN...")
    X_train_features = extract_features_batch(feature_extractor, X_train, batch_size=batch_size)
    X_val_features = extract_features_batch(feature_extractor, X_val, batch_size=batch_size)
    X_test_features = extract_features_batch(feature_extractor, X_test, batch_size=batch_size)

    # Save extracted features to disk for fast evaluation & experimentation
    features_cache_path = os.path.join(models_dir, "extracted_features.npz")
    np.savez_compressed(
        features_cache_path,
        train_feat=X_train_features,
        val_feat=X_val_features,
        test_feat=X_test_features,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )
    logger.info("Extracted features cached to %s", features_cache_path)

    # 6. Fit Standard Scaler on Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features)
    X_test_scaled = scaler.transform(X_test_features)

    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved feature scaler to %s", scaler_path)

    # 7. Tune SVM Classifier on Validation Dataset
    c_candidates = [0.1, 1.0, 5.0, 10.0]
    best_c = 1.0
    best_score = -1.0
    tuning_results = []

    logger.info("Tuning SVM hyperparameter C on validation set...")
    for c_val in c_candidates:
        candidate_svm = SVC(
            C=c_val,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=42
        )
        candidate_svm.fit(X_train_scaled, y_train)
        val_preds = candidate_svm.predict(X_val_scaled)
        val_bal_acc = balanced_accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="weighted")
        
        tuning_results.append({
            "C": c_val,
            "kernel": "rbf",
            "val_balanced_accuracy": float(val_bal_acc),
            "val_weighted_f1": float(val_f1)
        })
        logger.info("C=%.2f -> Val Balanced Acc: %.4f, Val F1: %.4f", c_val, val_bal_acc, val_f1)
        
        if val_bal_acc > best_score:
            best_score = val_bal_acc
            best_c = c_val

    logger.info("Best SVM parameter found: C=%.2f with Balanced Acc=%.4f", best_c, best_score)
    with open(os.path.join(reports_dir, "svm_tuning_results.json"), "w") as f:
        json.dump(tuning_results, f, indent=4)

    # 8. Train Final Optimal SVM with Probability Calibration
    logger.info("Training final calibrated SVM model with C=%.2f...", best_c)
    final_svm = SVC(
        C=best_c,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    final_svm.fit(X_train_scaled, y_train)

    svm_model_path = os.path.join(models_dir, "svm_model.joblib")
    joblib.dump(final_svm, svm_model_path)
    logger.info("Final SVM model successfully saved to %s", svm_model_path)

    return {
        "best_C": best_c,
        "val_score": best_score,
        "models_saved": [
            os.path.join(models_dir, "cnn_baseline.keras"),
            os.path.join(models_dir, "cnn_feature_extractor.keras"),
            svm_model_path,
            scaler_path
        ]
    }


if __name__ == "__main__":
    train_pipeline()
