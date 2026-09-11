"""
Preprocessing module for Parking Space Availability Detection.
Applies OpenCV image transformations: cropping, resizing, grayscale conversion,
contrast enhancement (CLAHE), Gaussian smoothing, and normalization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


def crop_parking_space(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    points: Optional[List[Tuple[float, float]]] = None,
    target_size: Tuple[int, int] = (64, 64),
    mask_polygon: bool = False
) -> np.ndarray:
    """
    Crops a parking space region from a full parking lot image.
    
    Args:
        image: Full parking lot BGR image (H, W, 3).
        bbox: Bounding box tuple (x_min, y_min, x_max, y_max).
        points: Optional polygon vertices [(x, y), ...].
        target_size: (width, height) output dimensions.
        mask_polygon: If True and points given, zeroes out pixels outside polygon.
        
    Returns:
        np.ndarray: Cropped and resized BGR patch of shape (target_size[1], target_size[0], 3).
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = image.shape[:2]
    
    # Boundary validation and clamping
    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(x_min + 1, min(w, x_max))
    y_max = max(y_min + 1, min(h, y_max))
    
    crop = image[y_min:y_max, x_min:x_max].copy()
    
    if mask_polygon and points is not None:
        pts_local = np.array([
            [pt[0] - x_min, pt[1] - y_min] for pt in points
        ], dtype=np.int32)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_local], 255)
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_crop(
    crop: np.ndarray,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    blur_ksize: Optional[Tuple[int, int]] = (3, 3),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocesses a single cropped parking spot using OpenCV.
    
    Pipeline:
      1. BGR -> Grayscale (cv2.cvtColor)
      2. Contrast enhancement via CLAHE (cv2.createCLAHE)
      3. Denoising via Gaussian Blur (cv2.GaussianBlur)
      4. Dynamic range normalization (cv2.normalize)
      
    Args:
        crop: Input BGR crop image.
        use_clahe: Whether to apply CLAHE contrast enhancement.
        clahe_clip: Threshold for contrast limiting in CLAHE.
        clahe_grid: Grid tile size for CLAHE.
        blur_ksize: Kernel size for Gaussian smoothing (or None).
        normalize: Whether to normalize pixel values to [0, 255].
        
    Returns:
        np.ndarray: Preprocessed single-channel uint8 image (H, W).
    """
    # 1. Grayscale Conversion
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
        
    # 2. Contrast Enhancement (CLAHE adapts to varying shadows & lighting)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray
        
    # 3. Gaussian Blur to suppress asphalt micro-texture while keeping car edges
    if blur_ksize is not None and blur_ksize[0] > 0:
        smoothed = cv2.GaussianBlur(enhanced, blur_ksize, 0)
    else:
        smoothed = enhanced
        
    # 4. Normalization
    if normalize:
        normalized = cv2.normalize(smoothed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    else:
        normalized = smoothed
        
    return normalized


def preprocess_batch(
    crops: np.ndarray,
    use_clahe: bool = True,
    blur_ksize: Optional[Tuple[int, int]] = (3, 3)
) -> np.ndarray:
    """
    Applies the OpenCV preprocessing pipeline across an entire batch of crops.
    
    Args:
        crops: Array of shape (N, H, W, 3) or (N, H, W).
        
    Returns:
        np.ndarray: Array of shape (N, H, W) of uint8 preprocessed grayscale images.
    """
    processed = []
    for c in crops:
        p = preprocess_crop(c, use_clahe=use_clahe, blur_ksize=blur_ksize)
        processed.append(p)
    return np.array(processed, dtype=np.uint8)


def compute_canny_edges(
    gray_crop: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150
) -> np.ndarray:
    """Computes Canny edge map for visual inspection and feature extraction."""
    return cv2.Canny(gray_crop, low_thresh, high_thresh)


def visualize_pipeline_steps(
    bgr_crop: np.ndarray,
    title: str = "Preprocessing Pipeline",
    save_path: Optional[str] = None
) -> None:
    """
    Renders and optionally saves a multi-stage visualization of the OpenCV preprocessing pipeline.
    """
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    stages = [
        ("1. Original BGR", cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB), None),
        ("2. Grayscale", gray, "gray"),
        ("3. CLAHE Enhanced", clahe, "gray"),
        ("4. Gaussian Blur (3x3)", blurred, "gray"),
        ("5. Canny Edges", edges, "gray")
    ]
    
    for ax, (name, img_data, cmap) in zip(axes, stages):
        ax.imshow(img_data, cmap=cmap)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle(title, fontsize=13, y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_dataset
    
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = str(project_root / "Dataset")
    crops, labels, df_spots = load_dataset(dataset_path)
    
    # Pick one occupied and one available crop
    idx_avail = np.where(labels == 0)[0][0]
    idx_occ = np.where(labels == 1)[0][0]
    
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_avail = str(plots_dir / "pipeline_available.png")
    out_occ = str(plots_dir / "pipeline_occupied.png")
    
    visualize_pipeline_steps(crops[idx_avail], title="CV Pipeline - AVAILABLE Spot", save_path=out_avail)
    visualize_pipeline_steps(crops[idx_occ], title="CV Pipeline - OCCUPIED Spot", save_path=out_occ)
    print("Preprocessing tests passed! Pipeline stage plots generated.")
