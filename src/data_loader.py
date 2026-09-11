"""
Data loader module for Parking Space Availability Detection.
Parses CVAT XML annotations and extracts individual parking space regions from parking lot imagery.
"""

import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from typing import List, Dict, Tuple, Optional


LABEL_MAPPING = {
    "free_parking_space": 0,             # AVAILABLE
    "not_free_parking_space": 1,         # OCCUPIED
    "partially_free_parking_space": 1     # OCCUPIED (Blocked/Unavailable for a new car)
}

CLASS_NAMES = {0: "AVAILABLE", 1: "OCCUPIED"}


def parse_xml_annotations(xml_path: str) -> pd.DataFrame:
    """
    Parses CVAT 1.1 XML file containing parking space polygon annotations.
    
    Args:
        xml_path: Path to annotations.xml file.
        
    Returns:
        pd.DataFrame: DataFrame containing image names, sizes, polygon points, and labels.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Annotations file not found at: {xml_path}")
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    records = []
    spot_counter = 0
    
    for img_elem in root.findall("image"):
        img_id = int(img_elem.get("id"))
        img_name = img_elem.get("name")
        width = int(img_elem.get("width"))
        height = int(img_elem.get("height"))
        
        for poly in img_elem.findall("polygon"):
            raw_label = poly.get("label")
            points_str = poly.get("points")
            
            # Parse coordinate string: "x1,y1;x2,y2;..."
            pts = []
            for pt in points_str.strip().split(";"):
                if pt.strip():
                    x, y = map(float, pt.split(","))
                    pts.append((x, y))
            
            pts_arr = np.array(pts, dtype=np.float32)
            x_min, y_min = np.min(pts_arr, axis=0)
            x_max, y_max = np.max(pts_arr, axis=0)
            
            # Clamp coordinates to image dimensions
            x_min = max(0, int(np.floor(x_min)))
            y_min = max(0, int(np.floor(y_min)))
            x_max = min(width, int(np.ceil(x_max)))
            y_max = min(height, int(np.ceil(y_max)))
            
            # Enforce minimum dimension of 2px
            if x_max <= x_min:
                x_max = min(width, x_min + 2)
            if y_max <= y_min:
                y_max = min(height, y_min + 2)
            
            binary_label = LABEL_MAPPING.get(raw_label, 1)
            
            records.append({
                "spot_id": spot_counter,
                "image_id": img_id,
                "image_name": img_name,
                "raw_label": raw_label,
                "label": binary_label,
                "label_name": CLASS_NAMES[binary_label],
                "points": pts,
                "bbox": (x_min, y_min, x_max, y_max),
                "width": x_max - x_min,
                "height": y_max - y_min,
                "img_width": width,
                "img_height": height
            })
            spot_counter += 1
            
    df = pd.DataFrame(records)
    return df


def load_dataset(
    dataset_dir: str,
    target_size: Tuple[int, int] = (64, 64),
    apply_polygon_mask: bool = False,
    exclude_partially_free: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads all parking lot images, extracts parking space patches, and prepares arrays for ML.
    
    Args:
        dataset_dir: Path to dataset root directory (containing 'images' and 'annotations.xml').
        target_size: (width, height) to resize extracted crops.
        apply_polygon_mask: If True, zeroes out pixels outside the polygon boundary within bbox.
        exclude_partially_free: If True, discards the 6 partially_free spots.
        
    Returns:
        crops: np.ndarray of shape (N, height, width, 3) with BGR crop images.
        labels: np.ndarray of shape (N,) with integer class labels (0 or 1).
        df_spots: pd.DataFrame with metadata for all extracted spots.
    """
    xml_path = os.path.join(dataset_dir, "annotations.xml")
    images_dir = os.path.join(dataset_dir, "images")
    
    df_spots = parse_xml_annotations(xml_path)
    
    if exclude_partially_free:
        df_spots = df_spots[df_spots["raw_label"] != "partially_free_parking_space"].reset_index(drop=True)
    
    # Cache loaded full images
    unique_images = df_spots["image_name"].unique()
    image_cache = {}
    for img_rel_path in unique_images:
        base_name = os.path.basename(img_rel_path)
        full_img_path = os.path.join(images_dir, base_name)
        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image file not found: {full_img_path}")
        img = cv2.imread(full_img_path)
        if img is None:
            raise ValueError(f"Failed to read image with OpenCV: {full_img_path}")
        image_cache[img_rel_path] = img
        
    crop_list = []
    valid_indices = []
    
    for idx, row in df_spots.iterrows():
        img = image_cache[row["image_name"]]
        x_min, y_min, x_max, y_max = row["bbox"]
        crop = img[y_min:y_max, x_min:x_max].copy()
        
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
            
        if apply_polygon_mask:
            # Shift polygon coordinates to crop local origin
            pts_local = np.array([
                [pt[0] - x_min, pt[1] - y_min] for pt in row["points"]
            ], dtype=np.int32)
            
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts_local], 255)
            crop = cv2.bitwise_and(crop, crop, mask=mask)
            
        # Resize to standardized target resolution
        resized_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        crop_list.append(resized_crop)
        valid_indices.append(idx)
        
    df_valid = df_spots.iloc[valid_indices].reset_index(drop=True)
    crops = np.array(crop_list, dtype=np.uint8)
    labels = df_valid["label"].to_numpy(dtype=np.int64)
    
    return crops, labels, df_valid


def split_data(
    crops: np.ndarray,
    labels: np.ndarray,
    df_spots: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "stratified"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Splits crops and labels into training and testing partitions.
    
    Strategies:
      - 'stratified': Random stratified split preserving class distribution at spot level.
      - 'group': Group split by image_id to test generalization to completely unseen parking scenes.
    """
    if split_strategy == "stratified":
        idx_train, idx_test = train_test_split(
            np.arange(len(labels)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
    elif split_strategy == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df_spots["image_id"].to_numpy()
        idx_train, idx_test = next(gss.split(crops, labels, groups=groups))
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}. Choose 'stratified' or 'group'.")
        
    X_train_crops = crops[idx_train]
    y_train = labels[idx_train]
    df_train = df_spots.iloc[idx_train].reset_index(drop=True)
    
    X_test_crops = crops[idx_test]
    y_test = labels[idx_test]
    df_test = df_spots.iloc[idx_test].reset_index(drop=True)
    
    if split_strategy == "group":
        train_groups = set(df_train["image_id"].unique())
        test_groups = set(df_test["image_id"].unique())
        assert train_groups.isdisjoint(test_groups), "Data leakage detected: scenes overlap across group train and test partitions!"
    
    return X_train_crops, X_test_crops, y_train, y_test, df_train, df_test


if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = str(project_root / "Dataset")
    
    crops, labels, df_spots = load_dataset(dataset_path)
    print(f"Loaded {len(crops)} crops with shape {crops.shape}")
    print(f"Class distribution: {dict(pd.Series(labels).value_counts())}")
    
    X_tr, X_te, y_tr, y_te, _, _ = split_data(crops, labels, df_spots, split_strategy="stratified")
    print(f"Stratified split -> Train: {len(X_tr)} (Available: {sum(y_tr==0)}, Occupied: {sum(y_tr==1)}) | Test: {len(X_te)} (Available: {sum(y_te==0)}, Occupied: {sum(y_te==1)})")
    
    X_tr_g, X_te_g, y_tr_g, y_te_g, df_tr_g, df_te_g = split_data(crops, labels, df_spots, split_strategy="group")
    print(f"Group split -> Train: {len(X_tr_g)} (Scenes: {df_tr_g['image_id'].nunique()}) | Test: {len(X_te_g)} (Scenes: {df_te_g['image_id'].nunique()})")
