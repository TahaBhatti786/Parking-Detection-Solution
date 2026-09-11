"""
Feature extraction module for Parking Space Availability Detection.
Extracts classical Computer Vision features:
  - Histogram of Oriented Gradients (HOG) (Dalal-Triggs standard)
  - Edge density via cv2.Canny
  - Intensity statistics (mean, standard deviation)
  - Color statistics (channel means and standard deviations)
  - Structural complexity via cv2.Laplacian variance
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """
    Classical Computer Vision feature extraction using OpenCV and HOG.
    Supports pure HOG, supplementary statistics, and combined feature vectors.
    """
    def __init__(
        self,
        mode: str = "hog_only",
        win_size: Tuple[int, int] = (64, 64),
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys"
    ):
        """
        Initializes HOG and CV feature extraction configuration.
        
        Args:
            mode: Feature set ('hog_only', 'simple_only', or 'combined').
            win_size: Detection window size (width, height).
            orientations: Number of orientation histogram bins (standard = 9).
            pixels_per_cell: Size of cell in pixels (standard = 8x8).
            cells_per_block: Number of cells per block (standard = 2x2).
            block_norm: Block normalization method (standard = 'L2-Hys').
        """
        self.mode = mode
        self.win_size = win_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_hog(self, gray_img: np.ndarray, return_image: bool = False):
        """
        Computes HOG descriptor from a grayscale image patch.
        Uses Dalal & Triggs formulation (9 orientation bins, 8x8 cells, 2x2 blocks).
        """
        if gray_img.shape[:2] != (self.win_size[1], self.win_size[0]):
            gray_img = cv2.resize(gray_img, self.win_size, interpolation=cv2.INTER_AREA)
            
        if return_image:
            features, hog_image = hog(
                gray_img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=True
            )
            return features, hog_image
        else:
            features = hog(
                gray_img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False
            )
            return features

    def extract_simple_features(self, bgr_img: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
        """
        Computes lightweight classical CV features:
          1. Edge density (cv2.Canny edge pixel ratio)
          2. Mean intensity
          3. Standard deviation of intensity
          4. Laplacian variance (structural high-frequency content via cv2.Laplacian)
          5-10. Color channel means and standard deviations (B, G, R)
        """
        h, w = gray_img.shape[:2]
        total_pixels = float(h * w)
        
        # 1. Edge density via OpenCV Canny
        edges = cv2.Canny(gray_img, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / total_pixels
        
        # 2-3. Grayscale intensity distribution
        mean_val = float(np.mean(gray_img)) / 255.0
        std_val = float(np.std(gray_img)) / 255.0
        
        # 4. Laplacian variance (OpenCV)
        lap_var = float(cv2.Laplacian(gray_img, cv2.CV_64F).var()) / 1000.0
        
        # 5-10. Color statistics across 3 channels
        b_mean = float(np.mean(bgr_img[:, :, 0])) / 255.0
        g_mean = float(np.mean(bgr_img[:, :, 1])) / 255.0
        r_mean = float(np.mean(bgr_img[:, :, 2])) / 255.0
        b_std = float(np.std(bgr_img[:, :, 0])) / 255.0
        g_std = float(np.std(bgr_img[:, :, 1])) / 255.0
        r_std = float(np.std(bgr_img[:, :, 2])) / 255.0
        
        return np.array([
            edge_density, mean_val, std_val, lap_var,
            b_mean, g_mean, r_mean, b_std, g_std, r_std
        ], dtype=np.float32)

    def extract_single(self, bgr_crop: np.ndarray, gray_preprocessed: np.ndarray) -> np.ndarray:
        """Extracts feature vector for a single crop based on the selected mode."""
        if self.mode == "hog_only":
            return self.extract_hog(gray_preprocessed)
        elif self.mode == "simple_only":
            return self.extract_simple_features(bgr_crop, gray_preprocessed)
        elif self.mode == "combined":
            hog_feats = self.extract_hog(gray_preprocessed)
            simple_feats = self.extract_simple_features(bgr_crop, gray_preprocessed)
            return np.concatenate([hog_feats, simple_feats])
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'hog_only', 'simple_only', or 'combined'.")

    def extract_batch(
        self,
        bgr_crops: np.ndarray,
        gray_preprocessed: np.ndarray
    ) -> np.ndarray:
        """
        Extracts feature vectors for an entire batch of parking spot crops.
        """
        features_list = []
        for i in range(len(bgr_crops)):
            feat = self.extract_single(bgr_crops[i], gray_preprocessed[i])
            features_list.append(feat)
        return np.array(features_list, dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits StandardScaler on training features and returns scaled features."""
        self.is_fitted = True
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms features using fitted StandardScaler."""
        if not self.is_fitted:
            raise RuntimeError("StandardScaler must be fitted using fit_transform before calling transform.")
        return self.scaler.transform(X)


def visualize_hog_comparison(
    crops: np.ndarray,
    gray: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Renders side-by-side comparison of original crops and their HOG gradient representations
    for both AVAILABLE and OCCUPIED parking spaces.
    """
    extractor = FeatureExtractor()
    idx_avail = np.where(labels == 0)[0][:2]
    idx_occ = np.where(labels == 1)[0][:2]
    indices = [idx_avail[0], idx_avail[1], idx_occ[0], idx_occ[1]]
    titles = [
        "Available Spot 1", "Available Spot 2",
        "Occupied Spot 1", "Occupied Spot 2"
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    for col, (idx, title) in enumerate(zip(indices, titles)):
        bgr = crops[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray_im = gray[idx]
        _, hog_img = extractor.extract_hog(gray_im, return_image=True)
        
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{title}\n(RGB Patch)", fontsize=10, fontweight="bold")
        axes[0, col].axis("off")
        
        axes[1, col].imshow(hog_img, cmap="inferno")
        axes[1, col].set_title("HOG Gradients", fontsize=10, fontweight="bold")
        axes[1, col].axis("off")
        
    plt.suptitle("Histogram of Oriented Gradients (HOG) Visual Signatures", fontsize=13, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_distributions(
    bgr_crops: np.ndarray,
    gray_preprocessed: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plots comparative distributions of classical CV features between Available and Occupied spots.
    """
    extractor = FeatureExtractor(mode="simple_only")
    features = extractor.extract_batch(bgr_crops, gray_preprocessed)
    
    feature_names = [
        "Edge Density", "Mean Intensity", "Intensity Std Dev", "Laplacian Variance",
        "Blue Mean", "Green Mean", "Red Mean", "Blue Std", "Green Std", "Red Std"
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    
    for i in range(10):
        vals_avail = features[labels == 0, i]
        vals_occ = features[labels == 1, i]
        
        axes[i].hist(vals_avail, bins=25, alpha=0.6, label="Available", color="#2ca02c", density=True)
        axes[i].hist(vals_occ, bins=25, alpha=0.6, label="Occupied", color="#d62728", density=True)
        axes[i].set_title(feature_names[i], fontsize=10, fontweight="bold")
        axes[i].tick_params(labelsize=8)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)
            
    plt.suptitle("Classical Computer Vision Feature Distributions by Class", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_dataset
    from preprocessing import preprocess_batch
    
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = str(project_root / "Dataset")
    crops, labels, df_spots = load_dataset(dataset_path)
    gray = preprocess_batch(crops)
    
    ext_hog = FeatureExtractor(mode="hog_only")
    feats_hog = ext_hog.extract_batch(crops, gray)
    print(f"HOG feature shape: {feats_hog.shape}")
    
    ext_comb = FeatureExtractor(mode="combined")
    feats_comb = ext_comb.extract_batch(crops, gray)
    print(f"Combined feature shape: {feats_comb.shape}")
    
    out_dir = project_root / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_hog_comparison(crops, gray, labels, save_path=str(out_dir / "hog_visualization.png"))
    plot_feature_distributions(crops, gray, labels, save_path=str(out_dir / "feature_distributions.png"))
    print("Feature extraction test passed! HOG visualizations and distribution plots generated.")
