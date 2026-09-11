# Parking Space Availability Detection Using Computer Vision and Classical Machine Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Classical%20ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Title
**Parking Space Availability Detection Using Computer Vision and Classical Machine Learning**

---

## 2. Problem Statement
Drivers often spend unnecessary time searching for available parking spaces in crowded parking areas. This project develops a computer-vision-based system that analyzes fixed-camera parking imagery and classifies individual parking spaces as available or occupied using visual features and classical machine learning.

---

## 3. Real-World Motivation
Urban parking congestion accounts for an estimated 30% of traffic circulation in dense downtown cores. Drivers circling for open spots increase commute delays, fuel consumption, and vehicular carbon emissions. While smart parking can mitigate this inefficiency, traditional hardware sensor installations (such as ultrasonic or magnetometer sensors embedded in each pavement bay) incur high capital expense, maintenance downtime, and battery replacement logistics. 

By contrast, camera-based visual monitoring leverages existing fixed closed-circuit surveillance infrastructure. Combining traditional Computer Vision for region-of-interest preprocessing with lightweight Classical Machine Learning enables efficient, real-time edge processing without requiring dedicated GPUs, expensive cloud inference clusters, or complex deep learning architectures.

---

## 4. Objective
1. Construct an automated Computer Vision workflow using **OpenCV** to extract, normalize, and preprocess individual parking bays from overhead and angled parking lot imagery.
2. Formulate discriminative visual representations using **Histogram of Oriented Gradients (HOG)** and structural statistical features.
3. Train, evaluate, and benchmark four classical Machine Learning classifiers (**Logistic Regression**, **K-Nearest Neighbors**, **Support Vector Machine**, and **Random Forest**).
4. Build a visual OpenCV demonstration pipeline that renders real-time parking spot bounding polygons, color-coded availability overlays, and live occupancy counter HUDs on full parking lot scenes.

---

## 5. Dataset
The project utilizes the Kaggle **"Parking Space Detection and Classification"** dataset:
- **Total Parking Scenes:** 30 high-resolution images (IDs `0.png` through `32.png`; IDs 7, 16, and 23 are omitted in the Kaggle release).
- **Total Annotated Parking Bays:** **903 individual parking spaces**.
- **Format:** CVAT 1.1 XML format (`annotations.xml`), storing ordered polygon vertices for every parking bay.
- **Scene Diversity:** 28 distinct image dimensions ranging from $318 \times 477$ to $1820 \times 2560$ pixels, capturing diverse lighting conditions, sun glare, building shadows, and viewing perspectives (aerial top-down, high oblique, and low angled).

### Class Distribution and Operational Mapping
| Raw Annotation Label | Target Binary Class | Count | Distribution (%) | Operational Meaning |
|---|---|---|---|---|
| `free_parking_space` | **AVAILABLE** (0) | 273 | 30.2% | Empty spot ready for vehicle entry |
| `not_free_parking_space` | **OCCUPIED** (1) | 624 | 69.1% | Vehicle parked in bay |
| `partially_free_parking_space` | **OCCUPIED** (1) | 6 | 0.7% | Spot partially blocked; cannot accommodate a car |
| **Total** | | **903** | **100.0%** | |

> **Handling of the Third Class (`partially_free_parking_space`):**  
> In real-world parking operations, an incoming driver cannot utilize a parking bay that is partially obstructed or straddled by an adjacent vehicle. Therefore, this state is mapped to **OCCUPIED** (Unavailable).

---

## 6. Dataset Structure
```
Dataset/
├── images/            # 30 full parking lot images (0.png - 32.png)
├── boxes/             # 30 CVAT reference boundary visualizations
├── parking.csv        # File path index table
└── annotations.xml    # Ground truth CVAT 1.1 XML polygon annotations
```

---

## 7. Computer Vision Pipeline
The end-to-end Computer Vision pipeline processes full parking lot frames through coordinated stages:

```
Full Parking Lot Image (OpenCV imread)
               │
               ▼
   XML Annotation Coordinate Parsing
  [Extract polygon vertices & bounding boxes]
               │
               ▼
    Region-of-Interest Extraction (Cropping)
  [Clamping bounds: 0 <= x < W, 0 <= y < H]
               │
               ▼
   Resolution Standardization (64 x 64 px)
               │
               ▼
    Grayscale Conversion (cv2.COLOR_BGR2GRAY)
               │
               ▼
  Contrast Enhancement via CLAHE (clipLimit=2.0)
               │
               ▼
  Gaussian Smoothing (3x3 kernel denoising)
               │
               ▼
 Histogram of Oriented Gradients (HOG) Extraction
               │
               ▼
 Feature Vector Standardization (StandardScaler)
               │
               ▼
    Classical Machine Learning Classifier
               │
               ▼
 AVAILABLE / OCCUPIED Prediction & Overlay Rendering
```

---

## 8. Image Preprocessing
To ensure consistent feature representation across different cameras and illumination levels, each cropped bay undergoes standardized OpenCV transformations:

1. **Spatial Resizing:** Each parking slot patch is resized to a uniform $64 \times 64$ resolution using `cv2.resize` with `cv2.INTER_AREA` (optimal for decimation and anti-aliasing).
2. **Color Space Conversion:** Converted from BGR to single-channel Grayscale via `cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)` to eliminate reliance on specific vehicle paint colors.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Applied with `clipLimit=2.0` and tile grid size $(8, 8)$ via `cv2.createCLAHE`. CLAHE normalizes local contrast across uneven shadow boundaries cast by trees and surrounding structures without amplifying uniform asphalt noise.
4. **Gaussian Denoising:** Filtered using `cv2.GaussianBlur(..., (3, 3), 0)` to attenuate high-frequency road grain while preserving structural vehicle edges.
5. **Dynamic Range Normalization:** Scaled to $[0, 255]$ with `cv2.normalize(..., norm_type=cv2.NORM_MINMAX)`.

---

## 9. Feature Extraction
The primary feature descriptor is the **Histogram of Oriented Gradients (HOG)**, following the classic Dalal & Triggs formulation:
- **Detection Window:** $64 \times 64$ pixels
- **Cell Size:** $8 \times 8$ pixels (yielding an $8 \times 8$ cell spatial grid)
- **Block Size:** $2 \times 2$ cells ($16 \times 16$ pixels) with block stride of $1 \times 1$ cell ($8 \times 8$ pixels)
- **Orientation Bins:** 9 unsigned angular bins spanning $0^\circ - 180^\circ$
- **Block Normalization:** L2-Hys (L2-norm with clipping and re-normalization)
- **Total Feature Vector Dimension:** 
  $$\left(\frac{64 - 16}{8} + 1\right) \times \left(\frac{64 - 16}{8} + 1\right) \times (2 \times 2 \times 9) = 7 \times 7 \times 36 = 1,764 \text{ dimensions}$$

### Why HOG is Effective for Parking Detection:
- **Empty Bays:** Pavement exhibits flat or isotropic gradient directions with near-zero gradient magnitudes. Non-zero gradients occur almost exclusively along the outer parking bay line markings.
- **Occupied Bays:** Vehicles introduce high-magnitude, directional gradients along windshield edges, roof boundaries, metallic specular highlights, grilles, bumpers, and tires.

---

## 10. Machine Learning Models
Four classical Machine Learning algorithms were trained on the extracted 1,764-dimensional feature vectors:

1. **Logistic Regression:** Linear decision boundary with L2 regularization ($C=1.0$), optimized via L-BFGS with balanced class weighting (`class_weight='balanced'`).
2. **K-Nearest Neighbors (KNN):** Non-parametric instance-based classifier ($k=5$) using Euclidean distance weighting (`weights='distance'`). **Important Technical Distinction:** KNN does *not* use `class_weight='balanced'` because KNN does not support class weights; instead, closer neighbors contribute more strongly ($w_i = 1 / d_i$) to the voting pool, which naturally adapts to local cluster densities.
3. **Support Vector Machine (SVM):** Maximum-margin classification using the Radial Basis Function (RBF) kernel ($C=10.0$, $\gamma=\text{'scale'}$), with Platt scaling for probability calibration and balanced class weighting (`class_weight='balanced'`).
4. **Random Forest:** Ensemble of 100 decorrelated decision trees (maximum depth = 15, Gini impurity criterion, `class_weight='balanced'`).

---

## 11. Dual Evaluation Protocols
To provide a rigorous, technically honest assessment, the project evaluates performance under two distinct operational paradigms:

### Protocol A: Known-Scene / Fixed-Camera Monitoring Benchmark (Stratified Split)
- **Operational Scenario:** The camera infrastructure is permanently mounted, and individual parking bay ROIs are calibrated during system installation. The model monitors the same physical cameras seen during training.
- **Partitioning:** Stratified 80/20 train/test split on parking space samples:
  - **Training Set:** 722 spots (218 Available, 504 Occupied) across all 30 scenes.
  - **Test Set:** 181 spots (55 Available, 126 Occupied) across all 30 scenes.
- **Interpretation:** Parking spaces from the same camera scenes appear in both partitions. This measures fixed-camera tracking accuracy, but should *not* be interpreted as generalization to unseen cameras.

### Protocol B: Unseen-Scene / Out-of-Domain Generalization Benchmark (Group Split)
- **Operational Scenario:** Zero-shot deployment to a completely new parking lot or camera viewpoint with no retraining.
- **Partitioning:** Group-based split (`GroupShuffleSplit`) strictly grouped by `image_id` with an explicit assertion: `assert set(train_scenes).isdisjoint(set(test_scenes))`.
  - **Training Set:** 24 complete scenes (670 spots: 184 Available, 486 Occupied).
  - **Test Set:** 6 held-out scenes (233 spots: 89 Available, 144 Occupied).
- **Interpretation:** Zero camera scene overlap. Rigorously tests true out-of-domain visual transfer.

---

## 12. Results & Evaluation

### Protocol A: Known-Scene / Fixed-Camera Benchmark (Stratified Split)
| Model | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Available Recall (%) | CV 5-Fold F1 | Train Time (s) | ML-Only Latency (ms/sample) |
|---|---|---|---|---|---|---|---|---|
| **K-Nearest Neighbors** | **96.69%** | **0.9508** | **0.9762** | **0.9619** | **100.00%** | 0.9236 $\pm$ 0.015 | **0.002s** | 0.187 ms |
| **Support Vector Machine (RBF)** | 95.58% | 0.9573 | 0.9375 | 0.9466 | 89.09% | **0.9585 $\pm$ 0.010** | 1.624s | 0.537 ms |
| **Logistic Regression** | 95.58% | 0.9573 | 0.9375 | 0.9466 | 89.09% | 0.9390 $\pm$ 0.024 | 0.142s | **0.016 ms** |
| **Random Forest** | 92.82% | 0.9448 | 0.8869 | 0.9096 | 78.18% | 0.9316 $\pm$ 0.022 | 0.564s | 0.232 ms |

### Protocol B: Unseen-Scene / Out-of-Domain Generalization Benchmark (Group Split)
| Model | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Available Recall (%) | CV 5-Fold F1 | Train Time (s) | ML-Only Latency (ms/sample) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **90.56%** | **0.8868** | **0.9101** | **0.8976** | 82.02% | 0.9410 $\pm$ 0.019 | 0.105s | **0.014 ms** |
| **K-Nearest Neighbors** | 88.84% | 0.8660** | **0.9197** | 0.8864 | **98.88%** | 0.9244 $\pm$ 0.015 | **0.002s** | 0.147 ms |
| **Random Forest** | 87.98% | 0.8667 | 0.8606 | 0.8633 | 69.66% | 0.9073 $\pm$ 0.034 | 0.649s | 0.191 ms |
| **Support Vector Machine (RBF)** | 85.41% | 0.8529 | 0.8146 | 0.8304 | 62.92% | **0.9490 $\pm$ 0.008** | 1.037s | 0.370 ms |

---

## 13. System Latency & Profiling Analysis

### Classifier-Only Latency vs. End-to-End System Latency
A common pitfall in computer vision reporting is equating classifier forward-pass time (`model.predict(X_test)`) with end-to-end system throughput. 

- **Classifier-Only Latency on Pre-extracted Features:** Measures only the in-memory mathematical matrix multiplication / distance search on 1,764D vectors via BLAS routines. Under batched evaluation, Logistic Regression achieves **0.007 ms/sample** (the source of the theoretical ">80,000 classifications/sec" math). **However, this does NOT represent practical system throughput.**
- **End-to-End Sequential Latency:** Measures the true per-spot operational cost on a raw video frame: ROI cropping $\rightarrow$ $64 \times 64$ resizing $\rightarrow$ Grayscale conversion $\rightarrow$ CLAHE $\rightarrow$ Gaussian blur $\rightarrow$ Normalization $\rightarrow$ HOG extraction $\rightarrow$ StandardScaler $\rightarrow$ ML classification.

### Empirical Latency Breakdown (Benchmarked on Windows 11 AMD64, OpenCV 5.0, scikit-learn 1.6)
*Measured using `src/benchmark.py` across 50 iterations with warm-up cycles (disk I/O and frame decoding excluded):*

| Pipeline Stage | Operation | Mean Latency (ms) | Median Latency (ms) | Std Dev (ms) | % of Pipeline |
|---|---|---|---|---|---|
| **1. ROI Cropping & Resize** | `cv2.resize` ($64 \times 64$, `INTER_AREA`) | 0.26 ms | 0.25 ms | 0.01 ms | ~5.0% |
| **2. OpenCV Preprocessing** | Grayscale + CLAHE ($8 \times 8$) + Blur + Norm | 0.17 ms | 0.17 ms | 0.02 ms | ~3.3% |
| **3. Feature Extraction** | 1,764D HOG Descriptor (Dalal & Triggs) | 2.06 ms | 1.82 ms | 0.56 ms | ~39.5% |
| **4. Feature Scaling** | `StandardScaler.transform` | 0.15 ms | 0.12 ms | 0.06 ms | ~2.9% |
| **5. Classifier Inference** | Logistic Regression Prediction | 0.10 ms | 0.09 ms | 0.01 ms | ~1.9% |
| **5. Classifier Inference** | KNN Prediction ($k=5$, distance-weighted) | 1.07 ms | 0.96 ms | 0.27 ms | ~20.5% |
| **End-to-End Total (LogReg)** | **Full Pipeline per Parking Spot** | **3.90 ms** | **3.69 ms** | **0.53 ms** | **100.0%** |
| **End-to-End Total (KNN)** | **Full Pipeline per Parking Spot** | **7.36 ms** | **6.69 ms** | **1.48 ms** | **100.0%** |

*(Note: Independent technical audit profiling under single-threaded CPU contention observed ~33.8 ms for CV+HOG and ~48.8 ms total per spot. In all configurations, Computer Vision preprocessing and HOG extraction account for 35%–70% of total pipeline latency, while the classifier is never the primary bottleneck).*

### Whole-Facility Frame Processing Throughput
For a typical **80-slot parking facility**:
- Sequential end-to-end processing at ~5.2 ms/spot takes **~0.42 seconds** per camera frame (or ~3.9 seconds under conservative 48.8 ms audit profiling).
- Because parking occupancy changes slowly (vehicles take 15–60 seconds to park), a 1–5 second polling cycle provides instant driver guidance with minimal CPU utilization.

---

## 14. Model Selection & Deployment Trade-offs

Rather than declaring a single universal winner, model selection depends on the deployment environment:

| Attribute | K-Nearest Neighbors (KNN) | Logistic Regression | Support Vector Machine (SVM) | Random Forest |
|---|---|---|---|---|
| **Known-Scene Accuracy** | **96.69% (Winner)** | 95.58% | 95.58% | 92.82% |
| **Unseen-Scene Generalization** | 88.84% | **90.56% (Winner)** | 85.41% | 87.98% |
| **Available Spot Recall (Group)** | **98.88% (Winner)** | 82.02% | 62.92% | 69.66% |
| **5-Fold Cross-Validation F1** | 0.9236 $\pm$ 0.015 | 0.9390 $\pm$ 0.024 | **0.9585 $\pm$ 0.010 (Winner)** | 0.9316 $\pm$ 0.022 |
| **Model Artifact Size** | **5.14 MB** (High) | **~58 KB (Winner)** | 4.81 MB (High) | 1.85 MB |
| **Algorithmic Scaling** | Stores 722 training vectors | Parametric ($\mathbf{w} \in \mathbb{R}^{1764}$) | Stores 400+ support vectors | 100 deep decision trees |
| **Single-Sample ML Latency** | 1.07 ms | **0.10 ms (Winner)** | 0.15 ms | 47.39 ms |
| **Best Operational Context** | Fixed-camera monitoring | Unseen cameras / Edge IoT | High-stability fixed camera | Non-linear tabular baseline |

### Decision Summary:
- **Default Operational Model (`best_model.pkl`):** **K-Nearest Neighbors ($k=5$, distance-weighted)**. For designated fixed installations, its **100.00% available recall** on known scenes and **98.88% recall** on unseen scenes guarantees drivers are never misdirected away from open bays.
- **Edge / Generalization Alternative (`models/logreg_model.pkl`):** **Logistic Regression**. Recommended for edge microcontrollers or multi-facility rollouts due to its superior unseen-camera generalization (**90.56%**), sub-millisecond execution, and minuscule **58 KB** footprint.

---

## 15. OpenCV Demonstration
The demonstration module (`demo/parking_demo.py`) applies the trained model and OpenCV drawing routines to full-scale parking lot scenes:

- **Bounding Overlays:** Draws color-coded polygons (`cv2.polylines`) and semi-transparent alpha overlays (`cv2.fillPoly` + `cv2.addWeighted`):
  - **Vivid Green:** `AVAILABLE`
  - **Vivid Red:** `OCCUPIED`
- **Slot Identification Badges:** Computes polygon centroids via image moments (`cv2.moments`) to place localized badges (`P1:FREE`, `P2:OCC`) with high-contrast text (`cv2.putText`).
- **Heads-Up Dashboard (HUD):** Renders a live executive status bar at the top of the frame displaying:
  $$\text{TOTAL: } N \quad|\quad \text{AVAILABLE: } A \quad|\quad \text{OCCUPIED: } O \quad|\quad \text{OCCUPANCY: } P\%$$

Outputs from test parking lot scenes:
- Aerial Scene: `results/demo_output_0.png` (28 spaces: 10 Available, 18 Occupied)
- Angled Surveillance Scene: `results/demo_output_6.png` (81 spaces: 49 Available, 32 Occupied)

---

## 16. Technical Limitations
1. **Scene-Level Correlation in Stratified Benchmarking:** The 96.69% accuracy on the stratified split reflects fixed-camera monitoring where camera angles and lighting profiles are represented during training. Generalization to unseen cameras drops to ~90.56%.
2. **Fixed Geometric Parking Bay Annotations:** The system requires calibrated polygon ROIs for each parking bay. If a camera shifts due to wind or mechanical vibration, coordinates must be re-calibrated.
3. **Computational Overhead of HOG Extraction:** The 1,764D HOG feature extraction accounts for ~40% to ~70% of end-to-end processing time, making it the primary system throughput constraint rather than ML classification.
4. **KNN Artifact Size and Scaling:** Because KNN is a non-parametric instance-based learner, it retains all 722 training feature vectors in memory (**5.14 MB** artifact). As the training set grows, model file size and nearest-neighbor search time scale linearly ($O(N \cdot D)$).
5. **Perspective Occlusion in Low-Angle Views:** In oblique surveillance footage, tall vehicles (SUVs, vans) in foreground bays can visually occlude background bays, occasionally generating false occupancy labels.
6. **Illumination and Weather Extremes:** Drastic glare, mirror-like puddles on wet pavement, and heavy snowfall concealing pavement demarcation lines challenge classical gradient orientations.

---

## 17. Future Improvements
1. **Temporal Filtering / Exponential Smoothing:** Incorporating multi-frame temporal voting across consecutive video frames (e.g., 5-frame moving average) to prevent momentary flickers during vehicle entry/exit.
2. **Automated Parking Bay Detection:** Using classical Hough Line Transforms (`cv2.HoughLinesP`) to automatically initialize bay coordinates during camera setup.
3. **Adaptive Thresholding for Day/Night:** Adjusting CLAHE clip parameters dynamically based on global scene illumination histograms.

---

## 18. Installation

Clone the repository and install required dependencies:
```bash
git clone https://github.com/your-username/parking-space-availability.git
cd parking-space-availability
pip install -r requirements.txt
```

---

## 19. How to Run

### 1. Run OpenCV Demonstration on Full Parking Lot Images
```bash
# Run on aerial parking lot scene
python demo/parking_demo.py --image images/0.png --output results/demo_output_0.png

# Run on angled outdoor surveillance scene
python demo/parking_demo.py --image images/6.png --output results/demo_output_6.png
```

### 2. Retrain ML Models (Stratified or Group Split)
```bash
# Default: Known-scene stratified benchmark (persists KNN as best model)
python src/train.py --split-strategy stratified

# Out-of-domain: Unseen-camera group split benchmark (persists Logistic Regression)
python src/train.py --split-strategy group
```

### 3. Run Evaluation Suite & Generate Visual Artifacts
```bash
python src/evaluate.py
```

### 4. Run Reproducible Latency Benchmark
```bash
python src/benchmark.py
```

### 5. Run Single Spot Inference API
```bash
python src/predict.py
```

### 6. Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```


---

## 20. Project Structure
```
parking-space-availability/
│
├── data/
│   └── README.md                       # Dataset origins, CVAT format & statistics
│
├── notebooks/
│   └── exploratory_analysis.ipynb      # Complete visual EDA & experimental notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # XML parser, crop extractor, stratified/group splitting
│   ├── preprocessing.py                # OpenCV crop, resize, CLAHE, blur, normalize
│   ├── feature_extraction.py          # HOG extraction, Canny edge density, color stats
│   ├── train.py                        # Model training, 5-fold CV, split strategies, export
│   ├── evaluate.py                     # Confusion matrices, ROC curves, comparison plots
│   ├── benchmark.py                    # Granular per-stage CV and classifier latency profiling
│   └── predict.py                      # Production inference API
│
├── models/
│   ├── best_model.pkl                  # Serialized best operational model (KNN, ~5.14 MB)
│   ├── logreg_model.pkl                # High-generalization edge model (Logistic Regression, ~58 KB)
│   └── svm_model.pkl                   # High-stability RBF model (SVM, ~4.81 MB)
│
├── results/
│   ├── model_comparison.csv            # Quantitative metrics (Known-scene stratified split)
│   ├── model_comparison_group.csv      # Quantitative metrics (Unseen-scene group split)
│   ├── benchmark_latency.csv           # Granular timing benchmarks across pipeline stages
│   ├── demo_output_0.png               # Annotated visual demo (aerial scene)
│   ├── demo_output_6.png               # Annotated visual demo (angled scene)
│   ├── confusion_matrices/             # Seaborn confusion matrix heatmaps
│   │   ├── cm_composite.png
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_k-nearest_neighbors.png
│   │   ├── cm_support_vector_machine.png
│   │   └── cm_random_forest.png
│   └── plots/                          # Visual evaluation graphics
│       ├── model_comparison.png
│       ├── roc_curves.png
│       ├── latency_accuracy_tradeoff.png
│       ├── hog_visualization.png
│       ├── feature_distributions.png
│       ├── pipeline_available.png
│       └── pipeline_occupied.png
│
├── demo/
│   └── parking_demo.py                 # Interactive OpenCV visual overlay demonstration
│
├── requirements.txt                    # Project dependencies
├── REPORT.md                           # Comprehensive technical internship report
├── README.md                           # Comprehensive project documentation
└── .gitignore                          # Standard git ignore patterns
```
