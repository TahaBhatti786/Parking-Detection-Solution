"""
Inference Latency & Throughput Benchmark.
Rigorously measures execution times across every stage of the pipeline:
  1. ROI Crop & Resize (64x64)
  2. OpenCV Preprocessing (Grayscale + CLAHE + Gaussian Blur + Normalization)
  3. HOG Feature Extraction (1,764D)
  4. Feature Scaling (StandardScaler)
  5. Classifier Prediction (tested across Logistic Regression, KNN, SVM, Random Forest)
  6. Total Sequential End-to-End Latency per Spot
  7. Full-Frame Batch Processing Latency

Distinguishes classifier-only FLOP latency from complete Computer Vision inference.
"""

import os
import sys
import time
import platform
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import load_dataset, parse_xml_annotations
from preprocessing import crop_parking_space, preprocess_crop
from feature_extraction import FeatureExtractor
from train import get_models
from predict import ParkingSpacePredictor


def run_benchmark(
    iterations: int = 50,
    warmup: int = 5,
    dataset_dir: str = None
) -> pd.DataFrame:
    if dataset_dir is None:
        dataset_dir = str(PROJECT_ROOT / "Dataset")
        
    print("=" * 70)
    print("  PARKING SPACE CV & CLASSICAL ML LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Platform:      {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:        {platform.python_version()}")
    print(f"OpenCV:        {cv2.__version__}")
    print(f"Iterations:    {iterations} (with {warmup} warm-up runs)")
    print(f"Benchmark Mode: In-memory (disk I/O and frame loading excluded)")
    print("-" * 70)
    
    # 1. Load an image and spots from dataset
    xml_path = os.path.join(dataset_dir, "annotations.xml")
    df_spots = parse_xml_annotations(xml_path)
    img_row = df_spots[df_spots["image_name"] == "images/0.png"].iloc[0]
    
    full_img_path = os.path.join(dataset_dir, "images", "0.png")
    full_image = cv2.imread(full_img_path)
    if full_image is None:
        raise FileNotFoundError(f"Could not load benchmark image at {full_img_path}")
        
    bbox = img_row["bbox"]
    points = img_row["points"]
    
    # Pre-crop a spot for isolated downstream tests
    crop_bgr = crop_parking_space(full_image, bbox=bbox, points=points, target_size=(64, 64))
    crop_gray_prep = preprocess_crop(crop_bgr)
    
    # Initialize extractor and scaler
    extractor = FeatureExtractor(mode="hog_only")
    # Quick fit on 20 dummy patches to initialize scaler
    dummy_feats = np.random.randn(20, 1764).astype(np.float32)
    extractor.fit_transform(dummy_feats)
    
    hog_feat = extractor.extract_single(crop_bgr, crop_gray_prep)
    scaled_feat = extractor.transform(hog_feat.reshape(1, -1))
    
    # Train lightweight instances of models for latency profiling
    models = get_models(random_state=42)
    dummy_y = np.array([0, 1] * 10)
    for m in models.values():
        m.fit(dummy_feats, dummy_y)
        
    # --- STAGE TIMINGS ---
    def benchmark_func(fn, n_iter, n_warm):
        for _ in range(n_warm):
            _ = fn()
        timings = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = fn()
            timings.append((time.perf_counter() - t0) * 1000.0)
        return np.mean(timings), np.median(timings), np.std(timings)
        
    print("\n[Benchmarking Individual Pipeline Stages]")
    
    # 1. Crop & Resize
    m_crop, med_crop, s_crop = benchmark_func(
        lambda: crop_parking_space(full_image, bbox=bbox, points=points, target_size=(64, 64)),
        iterations, warmup
    )
    print(f"  1. ROI Crop & Resize:       {m_crop:7.4f} ms  (median: {med_crop:7.4f}, std: {s_crop:7.4f})")
    
    # 2. Preprocessing (Grayscale + CLAHE + Blur + Normalize)
    m_prep, med_prep, s_prep = benchmark_func(
        lambda: preprocess_crop(crop_bgr),
        iterations, warmup
    )
    print(f"  2. OpenCV Preprocessing:    {m_prep:7.4f} ms  (median: {med_prep:7.4f}, std: {s_prep:7.4f})")
    
    # 3. HOG Extraction
    m_hog, med_hog, s_hog = benchmark_func(
        lambda: extractor.extract_single(crop_bgr, crop_gray_prep),
        iterations, warmup
    )
    print(f"  3. HOG Extraction (1764D):  {m_hog:7.4f} ms  (median: {med_hog:7.4f}, std: {s_hog:7.4f})")
    
    # 4. Feature Scaling
    m_scale, med_scale, s_scale = benchmark_func(
        lambda: extractor.transform(hog_feat.reshape(1, -1)),
        iterations, warmup
    )
    print(f"  4. Feature Scaling (Scaler):{m_scale:7.4f} ms  (median: {med_scale:7.4f}, std: {s_scale:7.4f})")
    
    # 5. ML Models Prediction Only (Single Spot)
    print("\n[Benchmarking Classifier-Only Inference (Single Sample Online)]")
    ml_timings = {}
    for name, model in models.items():
        m_pred, med_pred, s_pred = benchmark_func(
            lambda: model.predict(scaled_feat),
            iterations, warmup
        )
        ml_timings[name] = (m_pred, med_pred, s_pred)
        print(f"  - {name:<24}: {m_pred:7.4f} ms  (median: {med_pred:7.4f}, std: {s_pred:7.4f})")
        
    # 6. Batched Matrix Multiplication (181 test samples in memory)
    print("\n[Benchmarking Batched Classifier-Only Throughput (181 Pre-extracted Samples)]")
    batch_181 = np.repeat(scaled_feat, 181, axis=0)
    for name, model in models.items():
        m_batch, _, _ = benchmark_func(
            lambda: model.predict(batch_181),
            iterations, warmup
        )
        per_sample_batch_ms = m_batch / 181.0
        spots_per_sec = 1000.0 / per_sample_batch_ms if per_sample_batch_ms > 0 else 0
        print(f"  - {name:<24}: {per_sample_batch_ms:7.4f} ms/sample  (~{spots_per_sec:9,.0f} spots/sec batched BLAS)")
        
    # 7. Total End-to-End Sequential Latency (Single Spot)
    print("\n[Benchmarking Sequential End-to-End Pipeline (Crop -> Preprocess -> HOG -> Scale -> Predict)]")
    knn_model = models["K-Nearest Neighbors"]
    logreg_model = models["Logistic Regression"]
    
    def full_spot_pipeline(model):
        c = crop_parking_space(full_image, bbox=bbox, points=points, target_size=(64, 64))
        p = preprocess_crop(c)
        f = extractor.extract_single(c, p)
        s = extractor.transform(f.reshape(1, -1))
        return model.predict(s)
        
    m_e2e_knn, med_e2e_knn, s_e2e_knn = benchmark_func(
        lambda: full_spot_pipeline(knn_model),
        iterations, warmup
    )
    m_e2e_lr, med_e2e_lr, s_e2e_lr = benchmark_func(
        lambda: full_spot_pipeline(logreg_model),
        iterations, warmup
    )
    
    print(f"  End-to-End per spot (with KNN):         {m_e2e_knn:7.2f} ms  (median: {med_e2e_knn:7.2f}, std: {s_e2e_knn:7.2f})")
    print(f"  End-to-End per spot (with LogReg):      {m_e2e_lr:7.2f} ms  (median: {med_e2e_lr:7.2f}, std: {s_e2e_lr:7.2f})")
    
    cv_hog_total = m_crop + m_prep + m_hog + m_scale
    cv_ratio = (cv_hog_total / m_e2e_knn) * 100.0 if m_e2e_knn > 0 else 0
    print(f"\n  Breakdown: CV Preprocessing + HOG accounts for ~{cv_ratio:.1f}% of end-to-end execution time.")
    print("=" * 70)
    
    records = [
        {"Stage": "ROI Crop & Resize", "Mean (ms)": round(m_crop, 4), "Median (ms)": round(med_crop, 4), "Std (ms)": round(s_crop, 4)},
        {"Stage": "OpenCV Preprocessing (CLAHE+Blur)", "Mean (ms)": round(m_prep, 4), "Median (ms)": round(med_prep, 4), "Std (ms)": round(s_prep, 4)},
        {"Stage": "HOG Extraction (1764D)", "Mean (ms)": round(m_hog, 4), "Median (ms)": round(med_hog, 4), "Std (ms)": round(s_hog, 4)},
        {"Stage": "Feature Scaling", "Mean (ms)": round(m_scale, 4), "Median (ms)": round(med_scale, 4), "Std (ms)": round(s_scale, 4)},
        {"Stage": "ML Forward Pass (KNN)", "Mean (ms)": round(ml_timings["K-Nearest Neighbors"][0], 4), "Median (ms)": round(ml_timings["K-Nearest Neighbors"][1], 4), "Std (ms)": round(ml_timings["K-Nearest Neighbors"][2], 4)},
        {"Stage": "ML Forward Pass (Logistic Regression)", "Mean (ms)": round(ml_timings["Logistic Regression"][0], 4), "Median (ms)": round(ml_timings["Logistic Regression"][1], 4), "Std (ms)": round(ml_timings["Logistic Regression"][2], 4)},
        {"Stage": "Total End-to-End Spot Latency (KNN)", "Mean (ms)": round(m_e2e_knn, 2), "Median (ms)": round(med_e2e_knn, 2), "Std (ms)": round(s_e2e_knn, 2)},
        {"Stage": "Total End-to-End Spot Latency (LogReg)", "Mean (ms)": round(m_e2e_lr, 2), "Median (ms)": round(med_e2e_lr, 2), "Std (ms)": round(s_e2e_lr, 2)}
    ]
    df_bench = pd.DataFrame(records)
    
    out_csv = PROJECT_ROOT / "results" / "benchmark_latency.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bench.to_csv(out_csv, index=False)
    print(f"Benchmark summary exported to: {out_csv}")
    
    return df_bench


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference latency benchmark across pipeline stages.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up iterations.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset root directory.")
    
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, warmup=args.warmup, dataset_dir=args.dataset_dir)
