"""
Image preprocessing and dataset management module for Skin Cancer Detection.
Handles metadata loading, lesion-level leakage-free train/val/test splitting,
OpenCV-based image resizing, CLAHE contrast enhancement, normalization,
and data augmentation.
"""

import os
import glob
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES: Dict[str, str] = {
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma"
}

CLASS_TO_IDX: Dict[str, int] = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
IDX_TO_NAME: Dict[int, str] = {v: CLASS_NAMES[k] for k, v in CLASS_TO_IDX.items()}

SEVERITY_MAPPING: Dict[str, str] = {
    "nv": "Benign",
    "mel": "Malignant",
    "bkl": "Benign",
    "bcc": "Malignant",
    "akiec": "Pre-cancerous / Malignant",
    "vasc": "Benign",
    "df": "Benign"
}


def index_image_paths(search_dirs: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Search known directories for HAM10000 image files and index by image_id (without .jpg).
    """
    if search_dirs is None:
        search_dirs = [
            "HAM10000_images_part_1",
            "HAM10000_images_part_2",
            os.path.join("dataset", "raw"),
            os.path.join("dataset", "raw", "images"),
            os.path.join("dataset", "raw", "HAM10000_images_part_1"),
            os.path.join("dataset", "raw", "HAM10000_images_part_2"),
        ]

    image_path_map: Dict[str, str] = {}
    for d in search_dirs:
        if os.path.exists(d):
            for file_path in glob.glob(os.path.join(d, "*.jpg")):
                basename = os.path.splitext(os.path.basename(file_path))[0]
                if basename not in image_path_map:
                    image_path_map[basename] = os.path.abspath(file_path)

    logger.info("Indexed %d images across search directories.", len(image_path_map))
    return image_path_map


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
    to enhance skin lesion contrast while preserving true chromatic characteristics.
    Accepts both uint8 [0, 255] and float32 [0, 1] input.
    """
    if img_rgb.dtype != np.uint8:
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.clip(0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_single_image(
    image_path: str,
    target_size: Tuple[int, int] = (128, 128),
    enhance: bool = True,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Load an image from disk using OpenCV, resize, optionally enhance with CLAHE,
    and normalize pixel values to [0, 1].
    """
    if not os.path.exists(image_path):
        logger.error("Image file not found: %s", image_path)
        return None

    # Read image using OpenCV (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.error("Failed to decode image: %s", image_path)
        return None

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to consistent size
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Optional CLAHE enhancement
    if enhance:
        processed = apply_clahe(resized)
    else:
        processed = resized

    # Normalize to [0, 1] as float32
    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def augment_image_array(img: np.ndarray) -> np.ndarray:
    """
    Apply real-time randomized augmentation to an image:
    Random horizontal flip, vertical flip, and 90-degree rotations.
    """
    aug_img = img.copy()
    # Random horizontal flip
    if np.random.rand() > 0.5:
        aug_img = np.fliplr(aug_img)
    # Random vertical flip
    if np.random.rand() > 0.5:
        aug_img = np.flipud(aug_img)
    # Random 90 degree rotation
    rot_k = np.random.choice([0, 1, 2, 3])
    if rot_k > 0:
        aug_img = np.rot90(aug_img, k=rot_k)
    return np.ascontiguousarray(aug_img)


def create_leak_free_splits(
    metadata_df: pd.DataFrame,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train, validation, and test splits strictly grouped by lesion_id.
    This guarantees that all images corresponding to the same lesion belong
    exclusively to one split, avoiding any data leakage.
    """
    df = metadata_df.copy()
    # Group at the lesion level
    lesion_level = df.groupby("lesion_id").agg({
        "dx": "first",
        "image_id": "count"
    }).reset_index().rename(columns={"image_id": "image_count"})

    train_val_lesions, test_lesions = train_test_split(
        lesion_level,
        test_size=test_ratio,
        random_state=random_state,
        stratify=lesion_level["dx"]
    )

    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=relative_val_ratio,
        random_state=random_state,
        stratify=train_val_lesions["dx"]
    )

    train_ids = set(train_lesions["lesion_id"])
    val_ids = set(val_lesions["lesion_id"])
    test_ids = set(test_lesions["lesion_id"])

    # Double check zero leakage
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage detected between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage detected between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage detected between val and test!"

    train_df = df[df["lesion_id"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_ids)].copy().reset_index(drop=True)

    logger.info("Split completed: Train=%d images, Val=%d images, Test=%d images",
                len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def load_dataset_arrays(
    df: pd.DataFrame,
    image_path_map: Dict[str, str],
    target_size: Tuple[int, int] = (128, 128),
    enhance: bool = True,
    max_workers: int = 8
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess images referenced in dataframe using parallel threads.
    Returns:
        images: np.ndarray of shape (N, H, W, 3) in float32 [0, 1]
        labels: np.ndarray of shape (N,) integer class indices
        valid_image_ids: List of successfully loaded image IDs
    """
    image_ids = df["image_id"].tolist()
    dx_labels = [CLASS_TO_IDX[dx] for dx in df["dx"].tolist()]

    tasks = [(image_ids[i], image_path_map.get(image_ids[i]), dx_labels[i]) for i in range(len(image_ids))]

    def _worker(task):
        img_id, path, label = task
        if path is None:
            return None
        processed = preprocess_single_image(path, target_size=target_size, enhance=enhance, normalize=True)
        if processed is None:
            return None
        return img_id, processed, label

    valid_images = []
    valid_labels = []
    valid_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_worker, tasks):
            if result is not None:
                img_id, img_arr, lbl = result
                valid_ids.append(img_id)
                valid_images.append(img_arr)
                valid_labels.append(lbl)

    X = np.array(valid_images, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int64)
    return X, y, valid_ids


def save_processed_splits(
    output_dir: str = os.path.join("dataset", "processed"),
    metadata_csv: str = "HAM10000_metadata.csv",
    target_size: Tuple[int, int] = (128, 128),
    enhance: bool = True
):
    """
    Full pipeline to split metadata, preprocess all images, and save processed splits to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_map = index_image_paths()

    # Load metadata
    if not os.path.exists(metadata_csv):
        metadata_csv = os.path.join("dataset", "raw", "HAM10000_metadata.csv")
    df = pd.read_csv(metadata_csv)

    # Filter metadata to images that exist in map
    df = df[df["image_id"].isin(image_map)].copy().reset_index(drop=True)

    train_df, val_df, test_df = create_leak_free_splits(df)

    # Save split metadata for reference
    train_df.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)

    logger.info("Loading and preprocessing training set...")
    X_train, y_train, train_ids = load_dataset_arrays(train_df, image_map, target_size=target_size, enhance=enhance)
    logger.info("Loading and preprocessing validation set...")
    X_val, y_val, val_ids = load_dataset_arrays(val_df, image_map, target_size=target_size, enhance=enhance)
    logger.info("Loading and preprocessing test set...")
    X_test, y_test, test_ids = load_dataset_arrays(test_df, image_map, target_size=target_size, enhance=enhance)

    logger.info("Saving numpy arrays to %s...", output_dir)
    np.savez_compressed(
        os.path.join(output_dir, "preprocessed_data.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    # Save classes and mappings
    classes_info = {
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "idx_to_name": IDX_TO_NAME,
        "severity_mapping": SEVERITY_MAPPING,
        "image_shape": [target_size[0], target_size[1], 3]
    }
    with open(os.path.join(output_dir, "classes_info.json"), "w") as f:
        json.dump(classes_info, f, indent=4)

    logger.info("Dataset preprocessing and storage complete!")
    return {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "image_shape": X_train[0].shape
    }


if __name__ == "__main__":
    save_processed_splits()
