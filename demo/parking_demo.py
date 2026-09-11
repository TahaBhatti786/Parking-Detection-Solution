"""
OpenCV Parking Space Availability Demonstration.
Loads a parking lot image, extracts each parking space using dataset coordinates,
applies the identical OpenCV preprocessing and HOG feature extraction pipeline,
predicts availability using the trained classical ML model, and renders the result
with OpenCV drawing functions (cv2.polylines, cv2.rectangle, cv2.putText).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

# Ensure project root is in sys.path so 'src' package is cleanly resolved
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ParkingSpacePredictor
from src.data_loader import parse_xml_annotations


def run_parking_demo(
    image_name: str = "images/0.png",
    dataset_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show_window: bool = False
) -> None:
    """
    Executes the OpenCV visualization demo on a parking lot image.
    """
    # 1. Resolve absolute file paths cleanly
    if dataset_dir is None:
        dataset_dir = str(PROJECT_ROOT / "Dataset")
    else:
        d_obj = Path(dataset_dir)
        dataset_dir = str(d_obj if d_obj.is_absolute() else (PROJECT_ROOT / d_obj))
        
    if model_path is None:
        model_path = str(PROJECT_ROOT / "models" / "best_model.pkl")
    else:
        m_obj = Path(model_path)
        model_path = str(m_obj if m_obj.is_absolute() else (PROJECT_ROOT / m_obj))
        
    if output_path is None:
        output_path = str(PROJECT_ROOT / "results" / "demo_output.png")
    else:
        o_obj = Path(output_path)
        output_path = str(o_obj if o_obj.is_absolute() else (PROJECT_ROOT / o_obj))
        
    xml_path = os.path.join(dataset_dir, "annotations.xml")
    
    # Resolve target image file location
    img_name_obj = Path(image_name)
    if img_name_obj.is_absolute() and img_name_obj.exists():
        img_full_path = str(img_name_obj)
    elif (PROJECT_ROOT / img_name_obj).exists():
        img_full_path = str(PROJECT_ROOT / img_name_obj)
    elif os.path.exists(os.path.join(dataset_dir, "images", os.path.basename(image_name))):
        img_full_path = os.path.join(dataset_dir, "images", os.path.basename(image_name))
    elif os.path.exists(os.path.join(dataset_dir, image_name)):
        img_full_path = os.path.join(dataset_dir, image_name)
    else:
        img_full_path = os.path.join(dataset_dir, "images", os.path.basename(image_name))
    
    print("=" * 65)
    print("  OPENCV PARKING LOT AVAILABILITY DETECTION DEMO")
    print("=" * 65)
    print(f"Target Image:  {img_full_path}")
    print(f"Model Path:    {model_path}")
    print(f"Output Path:   {output_path}")
    
    # 2. Load the image using OpenCV
    image = cv2.imread(img_full_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {img_full_path}")
    h_img, w_img = image.shape[:2]
    print(f"Loaded image resolution: {w_img} x {h_img}")
    
    # 3. Parse XML annotations to get predefined coordinates for this scene
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Could not find annotations XML at: {xml_path}")
        
    df_annotations = parse_xml_annotations(xml_path)
    target_base = os.path.basename(image_name)
    spots_for_image = df_annotations[
        (df_annotations["image_name"] == image_name) |
        (df_annotations["image_name"].apply(os.path.basename) == target_base)
    ].to_dict(orient="records")
        
    print(f"Found {len(spots_for_image)} annotated parking spaces in this scene.")
    if len(spots_for_image) == 0:
        print(f"[WARNING] No annotations found matching '{image_name}'. Available scene IDs in XML include:")
        print("  ", list(df_annotations["image_name"].apply(os.path.basename).unique()[:10]), "...")
    
    # 4. Load the trained classical ML predictor
    predictor = ParkingSpacePredictor(model_bundle_path=model_path)
    print(f"Loaded Best Classifier: {predictor.model_name}")
    print(f"Feature Extraction:     {predictor.feature_mode} (HOG 1764D)")
    
    # 5. Predict on each spot
    predictions = predictor.predict_image(image, spots_for_image)
    
    # Calculate summary counts
    total_spaces = len(predictions)
    available_count = sum(1 for p in predictions if p["prediction"] == "AVAILABLE")
    occupied_count = total_spaces - available_count
    occupancy_rate = (occupied_count / total_spaces * 100) if total_spaces > 0 else 0
    
    print("\n--- Detection Results ---")
    print(f"Total Spaces: {total_spaces}")
    print(f"Available:    {available_count}")
    print(f"Occupied:     {occupied_count}")
    print(f"Occupancy:    {occupancy_rate:.1f}%\n")
    
    # 6. Render OpenCV visual overlay
    annotated = image.copy()
    overlay_layer = image.copy()
    
    # BGR Colors
    COLOR_AVAILABLE = (0, 220, 0)      # Vivid Green
    COLOR_OCCUPIED = (0, 0, 230)       # Vivid Red
    COLOR_BG_DARK = (15, 15, 15)       # Dark overlay for HUD
    
    # Draw parking spots
    for i, spot in enumerate(predictions):
        slot_id = f"P{i+1}"
        is_avail = (spot["prediction"] == "AVAILABLE")
        color = COLOR_AVAILABLE if is_avail else COLOR_OCCUPIED
        
        # Get polygon coordinates
        pts = np.array(spot["points"], dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw semi-transparent color fill
        cv2.fillPoly(overlay_layer, [pts], color=color)
        # Draw distinct polygon boundary line
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Calculate centroid for text badge placement
        moments = cv2.moments(pts)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            bbox = spot["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            
        # Draw slot badge with ID and status
        status_text = "FREE" if is_avail else "OCC"
        badge_text = f"{slot_id}:{status_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38 if w_img < 1000 else 0.50
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
        tx = max(0, cx - tw // 2)
        ty = max(th + 4, cy + th // 2)
        
        # Badge background rectangle
        cv2.rectangle(
            annotated,
            (tx - 3, ty - th - 3),
            (tx + tw + 3, ty + baseline),
            COLOR_BG_DARK,
            cv2.FILLED
        )
        cv2.rectangle(
            annotated,
            (tx - 3, ty - th - 3),
            (tx + tw + 3, ty + baseline),
            color,
            1
        )
        # Badge text
        cv2.putText(
            annotated, badge_text, (tx, ty - 1),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
        
    # Apply alpha blending for translucent fill (35% opacity)
    cv2.addWeighted(overlay_layer, 0.35, annotated, 0.65, 0, annotated)
    
    # 7. Draw Executive HUD Dashboard Banner at top of image
    hud_h = 68 if h_img > 600 else 52
    hud_overlay = annotated.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w_img, hud_h), COLOR_BG_DARK, cv2.FILLED)
    cv2.addWeighted(hud_overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.line(annotated, (0, hud_h), (w_img, hud_h), (80, 80, 80), 1)
    
    # Header row 1: System title
    font_main = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
        annotated,
        "PARKING SPACE AVAILABILITY DETECTOR  |  OPENCV + CLASSICAL ML",
        (16, 24 if hud_h > 60 else 20),
        font_main, 0.52 if w_img > 900 else 0.42, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    # Header row 2: Real-time statistics counters
    stats_text_1 = f"TOTAL: {total_spaces}   "
    stats_text_2 = f"AVAILABLE: {available_count}   "
    stats_text_3 = f"OCCUPIED: {occupied_count}   "
    stats_text_4 = f"OCCUPANCY: {occupancy_rate:.1f}%   "
    stats_text_5 = f"MODEL: {predictor.model_name} (HOG + L2)"
    
    y_stat = 50 if hud_h > 60 else 40
    font_stat = cv2.FONT_HERSHEY_SIMPLEX
    scale_stat = 0.44 if w_img > 900 else 0.36
    
    x_pos = 16
    cv2.putText(annotated, stats_text_1, (x_pos, y_stat), font_stat, scale_stat, (220, 220, 220), 1, cv2.LINE_AA)
    x_pos += cv2.getTextSize(stats_text_1, font_stat, scale_stat, 1)[0][0]
    
    cv2.putText(annotated, stats_text_2, (x_pos, y_stat), font_stat, scale_stat, (0, 255, 100), 2, cv2.LINE_AA)
    x_pos += cv2.getTextSize(stats_text_2, font_stat, scale_stat, 1)[0][0]
    
    cv2.putText(annotated, stats_text_3, (x_pos, y_stat), font_stat, scale_stat, (80, 80, 255), 2, cv2.LINE_AA)
    x_pos += cv2.getTextSize(stats_text_3, font_stat, scale_stat, 1)[0][0]
    
    cv2.putText(annotated, stats_text_4, (x_pos, y_stat), font_stat, scale_stat, (255, 200, 50), 1, cv2.LINE_AA)
    x_pos += cv2.getTextSize(stats_text_4, font_stat, scale_stat, 1)[0][0]
    
    if w_img > 950:
        cv2.putText(annotated, stats_text_5, (x_pos, y_stat), font_stat, scale_stat, (180, 180, 180), 1, cv2.LINE_AA)
        
    # 8. Save output visual
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"[SUCCESS] Annotated visual demo saved to: {output_path}")
    
    # 9. Optionally show window
    if show_window:
        try:
            cv2.imshow("Parking Space Availability Demo", annotated)
            print("Displaying demo window (press any key to close)...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Window display not available: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV Parking Space Availability Demo")
    parser.add_argument("--image", type=str, default="images/0.png", help="Image name relative to dataset (e.g., images/0.png)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to Dataset root directory (defaults to <project_root>/Dataset)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model bundle (defaults to <project_root>/models/best_model.pkl)")
    parser.add_argument("--output", type=str, default=None, help="Output annotated image path (defaults to <project_root>/results/demo_output.png)")
    parser.add_argument("--show", action="store_true", help="Display output window interactively")
    
    args = parser.parse_args()
    
    run_parking_demo(
        image_name=args.image,
        dataset_dir=args.dataset_dir,
        model_path=args.model,
        output_path=args.output,
        show_window=args.show
    )
