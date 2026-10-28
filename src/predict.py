"""
Inference and Prediction Pipeline for Skin Cancer Classification.
Loads trained CNN feature extractor and SVM classifier to classify unseen skin lesion images.
Includes input validation, OpenCV preprocessing, feature scaling, and confidence estimation.
"""

import os
import json
import logging
from typing import Union, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import joblib
import tensorflow as tf

from src.preprocessing import (
    apply_clahe,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IDX_TO_NAME,
    SEVERITY_MAPPING
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SkinCancerPredictor:
    """
    Complete inference pipeline integrating CNN Feature Extractor and SVM Classifier.
    """
    def __init__(
        self,
        extractor_path: str = os.path.join("models", "cnn_feature_extractor.keras"),
        svm_path: str = os.path.join("models", "svm_model.joblib"),
        scaler_path: str = os.path.join("models", "scaler.joblib"),
        target_size: tuple = (128, 128)
    ):
        self.target_size = target_size
        self.extractor_path = extractor_path
        self.svm_path = svm_path
        self.scaler_path = scaler_path
        self.extractor = None
        self.svm = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """Load and cache CNN feature extractor, SVM model, and scaler."""
        if not os.path.exists(self.extractor_path):
            raise FileNotFoundError(f"Feature extractor not found: {self.extractor_path}")
        if not os.path.exists(self.svm_path):
            raise FileNotFoundError(f"SVM model not found: {self.svm_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")

        logger.info("Loading CNN feature extractor from %s...", self.extractor_path)
        self.extractor = tf.keras.models.load_model(self.extractor_path, compile=False)
        
        logger.info("Loading SVM classifier from %s...", self.svm_path)
        self.svm = joblib.load(self.svm_path)

        logger.info("Loading feature scaler from %s...", self.scaler_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info("All model artifacts loaded successfully.")

    def validate_and_load_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Validate input format and convert to RGB numpy array.
        Raises ValueError if image is invalid, corrupt, or has incorrect channels.
        """
        img_rgb = None

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path does not exist: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"OpenCV could not decode image file: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode image from byte buffer.")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))

        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

            if arr.ndim == 2:
                img_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3:
                if arr.shape[2] == 4:
                    img_rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                elif arr.shape[2] == 3:
                    img_rgb = arr
                else:
                    raise ValueError(f"Unsupported channel dimension: {arr.shape[2]}")
            else:
                raise ValueError(f"Invalid array dimensions: {arr.ndim}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if img_rgb is None or img_rgb.size == 0:
            raise ValueError("Empty or invalid image data received.")

        if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
            raise ValueError(f"Image dimensions too small: {img_rgb.shape[:2]}. Minimum required is 10x10.")

        return img_rgb

    def preprocess_image(self, img_rgb: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Resize image to (128, 128), apply CLAHE enhancement, and normalize to [0, 1].
        """
        resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        if enhance:
            enhanced = apply_clahe(resized)
        else:
            enhanced = resized
        normalized = enhanced.astype(np.float32) / 255.0
        return normalized

    def predict(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        enhance: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end inference:
        1. Validate & load image.
        2. Preprocess (CLAHE, Resize 128x128, Normalize).
        3. Extract CNN feature representation.
        4. Standardize features.
        5. Predict class and class probabilities with SVM.
        """
        try:
            img_rgb = self.validate_and_load_image(image_input)
            processed_img = self.preprocess_image(img_rgb, enhance=enhance)

            # Add batch dimension: (1, 128, 128, 3)
            batch_img = np.expand_dims(processed_img, axis=0)

            # Feature extraction
            feature_vector = self.extractor.predict(batch_img, verbose=0)

            # Scaling
            scaled_features = self.scaler.transform(feature_vector)

            # SVM classification & probabilities
            pred_idx = int(self.svm.predict(scaled_features)[0])
            probabilities = self.svm.predict_proba(scaled_features)[0]

            pred_code = IDX_TO_CLASS[pred_idx]
            pred_name = IDX_TO_NAME[pred_idx]
            severity = SEVERITY_MAPPING.get(pred_code, "Unknown")
            confidence = float(probabilities[pred_idx])

            # Class-wise probability breakdown
            breakdown = {}
            for i, prob in enumerate(probabilities):
                c_code = IDX_TO_CLASS[i]
                c_name = IDX_TO_NAME[i]
                breakdown[f"{c_name} ({c_code})"] = round(float(prob) * 100, 2)

            # Sort probabilities descending
            sorted_breakdown = dict(sorted(breakdown.items(), key=lambda item: item[1], reverse=True))

            # Risk guidance
            if severity == "Malignant":
                risk_level = "High Risk (Malignant lesion detected)"
                recommendation = "Urgent: Prompt clinical examination and biopsy by a board-certified dermatologist is advised."
            elif severity.startswith("Pre-cancerous"):
                risk_level = "Moderate Risk (Pre-cancerous lesion)"
                recommendation = "Dermatological evaluation recommended to evaluate preventative treatment."
            else:
                risk_level = "Low Risk (Benign characteristics)"
                recommendation = "Routine skin self-check. Monitor for changes in color, size, or symmetry (ABCDE criteria)."

            return {
                "status": "success",
                "predicted_code": pred_code,
                "predicted_name": pred_name,
                "confidence": round(confidence * 100, 2),
                "severity": severity,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "probabilities": sorted_breakdown,
                "preprocessed_image": (processed_img * 255.0).astype(np.uint8),
                "disclaimer": "Educational research prototype only. Not intended for primary clinical diagnosis."
            }

        except Exception as e:
            logger.error("Prediction error: %s", str(e), exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "disclaimer": "Educational research prototype only."
            }


_PREDICTOR_INSTANCE: Optional[SkinCancerPredictor] = None

def get_predictor() -> SkinCancerPredictor:
    """Singleton getter for the SkinCancerPredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = SkinCancerPredictor()
    return _PREDICTOR_INSTANCE


def predict_skin_lesion(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
    """Convenience functional wrapper for inference."""
    predictor = get_predictor()
    return predictor.predict(image_input)
