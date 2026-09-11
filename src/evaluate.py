"""
Evaluation module for Parking Space Availability Detection.
Generates:
  - Confusion Matrix heatmaps (Seaborn)
  - Comparative performance bar charts
  - ROC Curves
  - Latency vs Performance trade-off charts
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = str(PROJECT_ROOT / "Dataset")
DEFAULT_MODELS_DIR = str(PROJECT_ROOT / "models")
DEFAULT_RESULTS_DIR = str(PROJECT_ROOT / "results")
DEFAULT_CONFUSION_DIR = str(PROJECT_ROOT / "results" / "confusion_matrices")
DEFAULT_PLOTS_DIR = str(PROJECT_ROOT / "results" / "plots")


def plot_confusion_matrices(
    artifacts: Dict[str, Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    save_dir: str = DEFAULT_CONFUSION_DIR
) -> None:
    """
    Plots and saves individual and composite confusion matrix heatmaps.
    """
    if class_names is None:
        class_names = ["AVAILABLE", "OCCUPIED"]
        
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Composite 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    axes = axes.flatten()
    
    for i, (name, art) in enumerate(artifacts.items()):
        cm = art["cm"]
        # Individual plot
        plt.figure(figsize=(5, 4.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, annot_kws={"size": 13, "weight": "bold"}
        )
        plt.title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=10)
        plt.ylabel("Ground Truth", fontsize=10)
        plt.tight_layout()
        clean_name = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(save_dir, f"cm_{clean_name}.png"), dpi=200)
        plt.close()
        
        # Add to composite grid
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, annot_kws={"size": 12, "weight": "bold"}
        )
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted Label", fontsize=9)
        axes[i].set_ylabel("Ground Truth", fontsize=9)
        
    plt.figure(fig.number)
    plt.suptitle("Model Evaluation: Confusion Matrices", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm_composite.png"), dpi=200)
    plt.close()
    print(f"Confusion matrices saved to {save_dir}")


def plot_model_comparison_bar(
    df_results: pd.DataFrame,
    save_path: str = str(Path(DEFAULT_PLOTS_DIR) / "model_comparison.png")
) -> None:
    """
    Plots a professional grouped bar chart comparing the 4 classical ML models.
    """
    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]
    
    x = np.arange(len(df_results))
    width = 0.18
    
    plt.figure(figsize=(10, 5.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, (metric, color) in enumerate(zip(metrics, palette)):
        plt.bar(x + i * width, df_results[metric] * 100, width, label=metric, color=color, alpha=0.9)
        # Value labels above bars
        for idx, val in enumerate(df_results[metric]):
            plt.text(idx + i * width, val * 100 + 0.8, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=7.5, rotation=0)
            
    plt.xlabel("Classical Machine Learning Model", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Performance Comparison Across Classical ML Algorithms", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x + width * 1.5, df_results["Model"], fontsize=10)
    plt.ylim(0, 110)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Comparison bar plot saved to {save_path}")


def plot_roc_curves(
    artifacts: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    save_path: str = str(Path(DEFAULT_PLOTS_DIR) / "roc_curves.png")
) -> None:
    """
    Plots ROC curves for all models.
    """
    plt.figure(figsize=(7.5, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for (name, art), color in zip(artifacts.items(), colors):
        y_proba = art["y_proba"]
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        else:
            # For models without direct probabilities
            fpr, tpr, _ = roc_curve(y_test, art["y_pred"])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, linestyle="--", label=f"{name} (AUC = {roc_auc:.3f})")
            
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_latency_tradeoff(
    df_results: pd.DataFrame,
    save_path: str = str(Path(DEFAULT_PLOTS_DIR) / "latency_accuracy_tradeoff.png")
) -> None:
    """
    Plots Inference Latency (ms/sample) vs Macro F1 score trade-off.
    """
    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, row in df_results.iterrows():
        plt.scatter(row["Latency (ms/sample)"], row["F1 (Macro)"] * 100, s=250, color=colors[i], label=row["Model"], alpha=0.85, edgecolors="black", linewidth=1.5)
        plt.annotate(
            row["Model"],
            (row["Latency (ms/sample)"], row["F1 (Macro)"] * 100),
            textcoords="offset points",
            xytext=(10, -5),
            fontweight="bold",
            fontsize=9
        )
        
    plt.xlabel("Prediction Latency (ms / sample)", fontsize=10, fontweight="bold")
    plt.ylabel("Macro F1-Score (%)", fontsize=10, fontweight="bold")
    plt.title("Real-Time Feasibility: Latency vs. F1-Score Trade-off", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Latency trade-off plot saved to {save_path}")


if __name__ == "__main__":
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        
    try:
        from src.train import train_and_evaluate_all
        from src.data_loader import load_dataset, split_data
    except ImportError:
        from train import train_and_evaluate_all
        from data_loader import load_dataset, split_data
    
    parser = argparse.ArgumentParser(description="Evaluate classical ML models and generate diagnostic plots.")
    parser.add_argument("--split-strategy", type=str, default="stratified", choices=["stratified", "group"],
                        help="Data splitting protocol.")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to Dataset root.")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Path to models directory.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Path to results directory.")
    
    args = parser.parse_args()
    
    df_res, artifacts, best_name = train_and_evaluate_all(
        dataset_dir=args.dataset_dir,
        feature_mode="hog_only",
        split_strategy=args.split_strategy,
        save_models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    crops, labels, df_spots = load_dataset(args.dataset_dir)
    _, _, _, y_te, _, _ = split_data(crops, labels, df_spots, split_strategy=args.split_strategy)
    
    plot_confusion_matrices(artifacts, save_dir=os.path.join(args.results_dir, "confusion_matrices"))
    plot_model_comparison_bar(df_res, save_path=os.path.join(args.results_dir, "plots", "model_comparison.png"))
    plot_roc_curves(artifacts, y_te, save_path=os.path.join(args.results_dir, "plots", "roc_curves.png"))
    plot_latency_tradeoff(df_res, save_path=os.path.join(args.results_dir, "plots", "latency_accuracy_tradeoff.png"))
    print("\nAll evaluation artifacts successfully created!")
