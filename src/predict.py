"""
Inference module for Parking Space Availability Detection.
Loads the serialized best model artifact and performs inference on:
  - Individual parking spot crops
  - Full parking lot images with coordinate lists
"""

import os
import pickle
import sys
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

# Ensure src directory is in sys.path for direct or module execution
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess_crop, crop_parking_space


class ParkingSpacePredictor:
    """
    Inference engine using the trained classical ML model and OpenCV pipeline.
    """
    def __init__(self, model_bundle_path: Optional[str] = None):
        from pathlib import Path
        if model_bundle_path is None:
            project_root = Path(__file__).resolve().parent.parent
            model_bundle_path = str(project_root / "models" / "best_model.pkl")
        elif not os.path.exists(model_bundle_path):
            project_root = Path(__file__).resolve().parent.parent
            alt_path = project_root / model_bundle_path.replace("../", "")
            if alt_path.exists():
                model_bundle_path = str(alt_path)
            else:
                raise FileNotFoundError(f"Model file not found at {model_bundle_path}")
                
        with open(model_bundle_path, "rb") as f:
            self.bundle = pickle.load(f)
            
        self.model_name = self.bundle["model_name"]
        self.model = self.bundle["model"]
        self.extractor = self.bundle["extractor"]
        self.feature_mode = self.bundle.get("feature_mode", "hog_only")
        
        self.classes = {0: "AVAILABLE", 1: "OCCUPIED"}

    def predict_crop(self, bgr_crop: np.ndarray) -> Tuple[str, float]:
        """
        Runs preprocessing, feature extraction, and prediction on a single crop.
        
        Returns:
            (predicted_class_name, confidence)
        """
        # Ensure standard 64x64 resolution
        if bgr_crop.shape[:2] != (64, 64):
            bgr_crop = cv2.resize(bgr_crop, (64, 64), interpolation=cv2.INTER_AREA)
            
        gray_crop = preprocess_crop(bgr_crop)
        features = self.extractor.extract_single(bgr_crop, gray_crop)
        scaled_features = self.extractor.transform(features.reshape(1, -1))
        
        pred_label = int(self.model.predict(scaled_features)[0])
        class_name = self.classes[pred_label]
        
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(scaled_features)[0, pred_label])
        else:
            proba = 1.0
            
        return class_name, proba

    def predict_image(
        self,
        full_image: np.ndarray,
        spots: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predicts availability for all parking spaces in a full parking lot image.
        
        Args:
            full_image: BGR full parking lot image.
            spots: List of dicts, each with 'bbox' (x_min, y_min, x_max, y_max)
                   and optionally 'points' [(x, y), ...].
                   
        Returns:
            List of dicts enriched with 'prediction', 'confidence', and 'class_id'.
        """
        results = []
        for i, spot in enumerate(spots):
            bbox = spot["bbox"]
            points = spot.get("points", None)
            crop = crop_parking_space(full_image, bbox=bbox, points=points, target_size=(64, 64))
            class_name, conf = self.predict_crop(crop)
            
            res = dict(spot)
            res["prediction"] = class_name
            res["class_id"] = 0 if class_name == "AVAILABLE" else 1
            res["confidence"] = conf
            results.append(res)
            
        return results


if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_dataset
    
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = str(project_root / "Dataset")
    models_path = str(project_root / "models" / "best_model.pkl")
    
    predictor = ParkingSpacePredictor(model_bundle_path=models_path)
    print(f"Predictor initialized with best model: {predictor.model_name}")
    
    crops, labels, df_spots = load_dataset(dataset_path)
    sample_crop = crops[0]
    pred_label, conf = predictor.predict_crop(sample_crop)
    gt_label = "AVAILABLE" if labels[0] == 0 else "OCCUPIED"
    print(f"Sample 0 -> Ground Truth: {gt_label} | Prediction: {pred_label} (Confidence: {conf*100:.1f}%)")
