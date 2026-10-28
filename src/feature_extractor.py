"""
CNN Feature Extractor Module.
Builds, trains, and uses a Convolutional Neural Network (MobileNetV2 backbone)
for extracting discriminative feature representations from dermoscopic skin lesion images.
"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import MobileNetV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_cnn_classifier(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 7,
    learning_rate: float = 1e-4,
    fine_tune_layers: int = 20
) -> Tuple[models.Model, models.Model]:
    """
    Constructs a MobileNetV2-based CNN classification model and a corresponding
    feature extractor sub-model.
    
    Returns:
        full_model: End-to-end model with classification head.
        feature_extractor: Model outputting the 1280-dim feature vector from GlobalAveragePooling.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Pretrained MobileNetV2 backbone
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )
    
    # Freeze initial layers, fine-tune top layers
    if fine_tune_layers > 0:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    # Feature extraction layer
    x = base_model.output
    features = layers.GlobalAveragePooling2D(name="cnn_features")(x)
    features_norm = layers.BatchNormalization(name="features_bn")(features)
    
    # Classification head for representation learning
    x = layers.Dropout(0.3, name="drop1")(features_norm)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_latent")(x)
    x = layers.Dropout(0.2, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    full_model = models.Model(inputs=inputs, outputs=outputs, name="SkinCancer_CNN")
    feature_extractor = models.Model(inputs=inputs, outputs=features, name="SkinCancer_FeatureExtractor")
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    full_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    logger.info("CNN classification and feature extraction models built successfully.")
    return full_model, feature_extractor


def extract_features_batch(
    extractor: models.Model,
    X_data: np.ndarray,
    batch_size: int = 64
) -> np.ndarray:
    """
    Extract dense feature vectors from image arrays in batches.
    """
    logger.info("Extracting CNN features for %d images with batch size %d...", len(X_data), batch_size)
    features = extractor.predict(X_data, batch_size=batch_size, verbose=1)
    logger.info("Extracted features matrix shape: %s", features.shape)
    return features


def save_models(
    full_model: models.Model,
    feature_extractor: models.Model,
    models_dir: str = "models"
):
    """
    Save both full CNN model and feature extractor model to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    full_model_path = os.path.join(models_dir, "cnn_baseline.keras")
    extractor_path = os.path.join(models_dir, "cnn_feature_extractor.keras")
    
    full_model.save(full_model_path)
    feature_extractor.save(extractor_path)
    logger.info("Saved baseline CNN to %s and feature extractor to %s", full_model_path, extractor_path)


def load_feature_extractor(model_path: str = os.path.join("models", "cnn_feature_extractor.keras")) -> models.Model:
    """
    Load saved CNN feature extractor model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Feature extractor model not found at {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Feature extractor loaded successfully from %s", model_path)
    return model
