# Parking Space Dataset Documentation

## Dataset Source
- **Origin:** Kaggle - *"Parking Space Detection and Classification"*
- **Location in Repository:** `Dataset/` (at project root)
- **Git Tracking:** The `Dataset/` directory is intentionally excluded from Git tracking via `.gitignore` due to file size (~40.6 MB).
- **CVAT Version:** 1.1 XML format

## Downloading and Setup Instructions
1. Download the *"Parking Space Detection and Classification"* dataset from Kaggle.
2. Extract the archive into the `Dataset/` directory located at the repository root:
   ```
   parking-space-availability/
   ├── Dataset/
   │   ├── images/
   │   ├── boxes/
   │   ├── parking.csv
   │   └── annotations.xml
   ```
3. Verify that `Dataset/annotations.xml` and `Dataset/images/` exist prior to running training or the demo.

## Directory Structure
```
Dataset/
├── images/            # 30 high-resolution parking lot scenes (0.png - 32.png)
├── boxes/             # 30 ground truth visual overlay reference images
├── parking.csv        # Metadata mapping index between images and masks
└── annotations.xml    # CVAT XML containing all polygon annotations
```
*(Note: IDs 7, 16, and 23 are omitted in the original dataset release, resulting in 30 total scenes).*

## Key Dataset Statistics
- **Total Parking Scenes:** 30
- **Distinct Image Dimensions:** 28 resolutions ranging from $318 \times 477$ to $1820 \times 2560$ pixels.
- **Total Annotated Parking Slots:** 903
- **Slot Geometry:**
  - 4-point quadrilaterals: 848 (93.9%)
  - 5-point polygons: 50 (5.5%)
  - 6-point polygons: 4 (0.4%)
  - 3-point triangles: 1 (0.1%)

## Class Breakdown & Mapping Strategy
| Raw Annotation Label in XML | Target Binary Class | Count | Percentage | Operational Meaning |
|---|---|---|---|---|
| `free_parking_space` | **AVAILABLE** (0) | 273 | 30.2% | Empty spot ready for vehicle entry |
| `not_free_parking_space` | **OCCUPIED** (1) | 624 | 69.1% | Full vehicle present in spot |
| `partially_free_parking_space` | **OCCUPIED** (1) | 6 | 0.7% | Spot partially obstructed; cannot accommodate a car |
| **Total** | | **903** | **100.0%** | |

### Handling of `partially_free_parking_space`
The dataset includes 6 instances labeled `partially_free_parking_space` (0.66% of all annotations). In a real-world parking guidance system, a driver cannot park in an obstructed or partially blocked bay without risk of damage or illegal parking. Consequently, this class is treated as **OCCUPIED (Unavailable)**. The pipeline also supports excluding these 6 samples via `exclude_partially_free=True` without statistically impacting model performance.
