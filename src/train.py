"""
Training module for Parking Space Availability Detection.
Trains and compares 4 classical Machine Learning classifiers:
  1. Logistic Regression
  2. K-Nearest Neighbors (KNN)
  3. Support Vector Machine (SVM)
  4. Random Forest
Measures training time, prediction latency, and performs cross-validation.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import sys
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, split_data
from preprocessing import preprocess_batch
from feature_extraction import FeatureExtractor


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes classical Machine Learning classifiers.
    
    Note on class imbalance handling:
      - Logistic Regression, SVM, and Random Forest use class_weight="balanced",
        which scales penalties inversely proportional to class frequencies.
      - K-Nearest Neighbors in scikit-learn does not support a class_weight parameter;
        it utilizes inverse Euclidean distance weighting (weights="distance"),
        where closer neighbor spots cast proportionally higher votes.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2
        ),
        "Support Vector Machine": SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }


def train_and_evaluate_all(
    dataset_dir: Optional[str] = None,
    feature_mode: str = "hog_only",
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
    save_models_dir: Optional[str] = None,
    results_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Complete end-to-end training and benchmark pipeline across all 4 models.
    Supports both 'stratified' (known-scene spot monitoring) and 'group' (unseen-camera generalization) splits.
    """
    project_root = Path(__file__).resolve().parent.parent
    if dataset_dir is None:
        dataset_dir = str(project_root / "Dataset")
    if save_models_dir is None:
        save_models_dir = str(project_root / "models")
    if results_dir is None:
        results_dir = str(project_root / "results")

    print(f"\n=======================================================")
    print(f"Starting Training Pipeline | Feature: {feature_mode} | Split: {split_strategy}")
    print(f"=======================================================")
    
    # 1. Load raw parking lot dataset & extract crops
    print("Loading dataset and extracting crops...")
    crops, labels, df_spots = load_dataset(dataset_dir)
    print(f"Total parking space crops: {len(crops)}")
    print(f"Class distribution: AVAILABLE (0) = {sum(labels==0)}, OCCUPIED (1) = {sum(labels==1)}")
    
    # 2. Data Splitting
    print(f"Splitting data using '{split_strategy}' strategy...")
    X_tr_crops, X_te_crops, y_train, y_test, df_train, df_test = split_data(
        crops, labels, df_spots, test_size=test_size, random_state=random_state, split_strategy=split_strategy
    )
    print(f"Training set: {len(X_tr_crops)} samples (Avail: {sum(y_train==0)}, Occ: {sum(y_train==1)}) across {df_train['image_id'].nunique()} scenes")
    print(f"Test set:     {len(X_te_crops)} samples (Avail: {sum(y_test==0)}, Occ: {sum(y_test==1)}) across {df_test['image_id'].nunique()} scenes")
    
    # 3. OpenCV Preprocessing
    print("Applying OpenCV preprocessing (Grayscale + CLAHE + Gaussian Blur + Normalization)...")
    gray_tr = preprocess_batch(X_tr_crops)
    gray_te = preprocess_batch(X_te_crops)
    
    # 4. Feature Extraction
    print(f"Extracting visual features (mode: {feature_mode})...")
    extractor = FeatureExtractor(mode=feature_mode)
    
    t0_feat = time.perf_counter()
    X_train_raw = extractor.extract_batch(X_tr_crops, gray_tr)
    X_test_raw = extractor.extract_batch(X_te_crops, gray_te)
    feat_time = time.perf_counter() - t0_feat
    print(f"Extracted {X_train_raw.shape[1]} features per sample in {feat_time:.2f}s.")
    
    # Scale features using training statistics ONLY (prevent data leakage)
    X_train = extractor.fit_transform(X_train_raw)
    X_test = extractor.transform(X_test_raw)
    
    # 5. Train and Benchmark Each Model
    models = get_models(random_state=random_state)
    results = []
    trained_artifacts = {}
    best_f1 = -1.0
    best_model_name = ""
    
    os.makedirs(save_models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    
    for name, model in models.items():
        print(f"\n--- Benchmarking: {name} ---")
        
        # 5-fold Stratified Cross-Validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        
        # Measure Training Time
        t0_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0_train
        
        # Measure Classifier-Only Inference Time (matrix multiplication on pre-extracted features)
        t0_pred = time.perf_counter()
        y_pred = model.predict(X_test)
        total_pred_time = time.perf_counter() - t0_pred
        latency_ms = (total_pred_time / len(X_test)) * 1000.0  # ms per sample (ML-only)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
            
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        # Specific metrics for AVAILABLE (minority class 0)
        rec_avail = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        prec_avail = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_avail = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"  CV 5-Fold F1 (Macro):  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  Test Accuracy:         {acc*100:.2f}%")
        print(f"  Test Macro F1:         {f1:.4f} (Available F1: {f1_avail:.4f})")
        print(f"  Available Recall:      {rec_avail*100:.2f}% (Found {sum((y_test==0)&(y_pred==0))}/{sum(y_test==0)} free spots)")
        print(f"  Training Time:         {train_time:.4f}s")
        print(f"  ML-Only Latency:       {latency_ms:.4f} ms/sample (classifier only)")
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec, 4),
            "Recall (Macro)": round(rec, 4),
            "F1 (Macro)": round(f1, 4),
            "CV F1 Mean": round(cv_scores.mean(), 4),
            "CV F1 Std": round(cv_scores.std(), 4),
            "Available Precision": round(prec_avail, 4),
            "Available Recall": round(rec_avail, 4),
            "Available F1": round(f1_avail, 4),
            "Train Time (s)": round(train_time, 4),
            "Latency (ms/sample)": round(latency_ms, 4)
        })
        
        trained_artifacts[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "cm": cm
        }
        
        # Track best model by Macro F1
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    df_results = pd.DataFrame(results)
    
    # Save results to CSV
    csv_filename = "model_comparison.csv" if split_strategy == "stratified" else f"model_comparison_{split_strategy}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    df_results.to_csv(csv_path, index=False)
    print(f"\nModel comparison table saved to: {csv_path}")
    
    # Save the Best Model Pipeline artifact
    best_bundle = {
        "model_name": best_model_name,
        "model": trained_artifacts[best_model_name]["model"],
        "extractor": extractor,
        "feature_mode": feature_mode,
        "split_strategy": split_strategy,
        "metrics": df_results[df_results["Model"] == best_model_name].to_dict(orient="records")[0]
    }
    best_model_path = os.path.join(save_models_dir, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_bundle, f)
    print(f"Best model '{best_model_name}' saved to: {best_model_path}")
    
    # Also save auxiliary models for flexible deployment (e.g. lightweight LogReg ~30KB vs KNN 5.14MB)
    for aux_name, filename in [("Logistic Regression", "logreg_model.pkl"), ("Support Vector Machine", "svm_model.pkl")]:
        if aux_name in trained_artifacts:
            aux_bundle = {
                "model_name": aux_name,
                "model": trained_artifacts[aux_name]["model"],
                "extractor": extractor,
                "feature_mode": feature_mode,
                "split_strategy": split_strategy,
                "metrics": df_results[df_results["Model"] == aux_name].to_dict(orient="records")[0]
            }
            with open(os.path.join(save_models_dir, filename), "wb") as f:
                pickle.dump(aux_bundle, f)
                
    return df_results, trained_artifacts, best_model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classical ML models on parking spot features.")
    parser.add_argument("--split-strategy", type=str, default="stratified", choices=["stratified", "group"],
                        help="Data splitting protocol: 'stratified' (known-scene monitoring) or 'group' (unseen-camera generalization).")
    parser.add_argument("--feature-mode", type=str, default="hog_only", choices=["hog_only", "combined", "simple_only"],
                        help="Feature extraction mode.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to Dataset root directory.")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to save trained models.")
    parser.add_argument("--results-dir", type=str, default=None, help="Path to save benchmark results.")
    
    args = parser.parse_args()
    
    df_res, artifacts, best_name = train_and_evaluate_all(
        dataset_dir=args.dataset_dir,
        feature_mode=args.feature_mode,
        split_strategy=args.split_strategy,
        save_models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    print("\n" + "="*50)
    print(f"TRAINING COMPLETE. WINNER: {best_name}")
    print("="*50)
    print(df_res[["Model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "Train Time (s)", "Latency (ms/sample)"]])
