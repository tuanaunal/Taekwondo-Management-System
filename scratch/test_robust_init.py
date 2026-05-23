import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.evaluator import PerformanceEvaluator
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR
from main import analyze_video

# Patched separate_helmet_foot with robust coherent tracking
def robust_separate_helmet_foot(self, mask: np.ndarray, frame: np.ndarray, color: str) -> tuple:
    from src.config import MIN_CONTOUR_AREA, MAX_CONTOUR_AREA
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = mask.shape[:2]
    mid_y = h // 2

    helmet_mask = np.zeros_like(mask)
    foot_mask = np.zeros_like(mask)

    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect_ratio = bw / bh

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Optimized border noise filters
        if cx < 180 or cx > w - 180 or cy < 130 or cy > h - 140:
            continue

        valid_contours.append({
            "cnt": cnt,
            "area": area,
            "centroid": (cx, cy),
            "aspect_ratio": aspect_ratio
        })

    if not valid_contours:
        return helmet_mask, foot_mask

    prev_h = self.prev_centroids[f"{color}_helmet"]
    prev_f = self.prev_centroids[f"{color}_foot"]

    helmet_cnt = None
    foot_cnt = None

    # Track using history if available
    if prev_h is not None:
        # Match helmet to the closest valid contour within 250px
        candidates_h = []
        for c in valid_contours:
            dist = np.linalg.norm(np.array(c["centroid"]) - np.array(prev_h))
            if dist < 250:
                candidates_h.append((dist, c))
        if candidates_h:
            candidates_h.sort(key=lambda x: x[0])
            helmet_cnt = candidates_h[0][1]

        # Match foot
        remaining = [c for c in valid_contours if c is not helmet_cnt]
        if prev_f is not None:
            candidates_f = []
            for c in remaining:
                dist = np.linalg.norm(np.array(c["centroid"]) - np.array(prev_f))
                if dist < 300:
                    candidates_f.append((dist, c))
            if candidates_f:
                candidates_f.sort(key=lambda x: x[0])
                foot_cnt = candidates_f[0][1]
        else:
            # If foot was lost or not initialized, find a candidate below the helmet and horizontally close
            ref_h = helmet_cnt["centroid"] if helmet_cnt else prev_h
            candidates_f = []
            for c in remaining:
                dy = c["centroid"][1] - ref_h[1]
                dx = abs(c["centroid"][0] - ref_h[0])
                if dy >= 150 and dx <= 300:
                    candidates_f.append((dy, c))
            if candidates_f:
                # Prefer the lowest candidate within range
                candidates_f.sort(key=lambda x: x[0], reverse=True)
                foot_cnt = candidates_f[0][1]

    # Robust Initialization (No history or lost helmet tracking)
    if helmet_cnt is None:
        # Try to find a coherent pair
        best_pair = None
        best_score = -float("inf")
        
        for i in range(len(valid_contours)):
            for j in range(len(valid_contours)):
                if i == j:
                    continue
                c_h = valid_contours[i]
                c_f = valid_contours[j]
                
                dy = c_f["centroid"][1] - c_h["centroid"][1]
                dx = abs(c_f["centroid"][0] - c_h["centroid"][0])
                
                if 150 <= dy <= 600 and dx <= 250:
                    score = (c_h["area"] + c_f["area"]) - 2.0 * dx
                    if score > best_score:
                        best_score = score
                        best_pair = (c_h, c_f)
                        
        if best_pair:
            helmet_cnt, foot_cnt = best_pair
        else:
            # Fallback: just pick the highest contour as helmet
            valid_contours.sort(key=lambda x: x["centroid"][1])
            helmet_cnt = valid_contours[0]
            ref_h = helmet_cnt["centroid"]
            candidates_f = [
                c for c in valid_contours[1:]
                if (c["centroid"][1] - ref_h[1]) >= 150 and abs(c["centroid"][0] - ref_h[0]) <= 250
            ]
            if candidates_f:
                foot_cnt = candidates_f[-1]

    # Draw masks and update state
    if helmet_cnt is not None:
        cv2.drawContours(helmet_mask, [helmet_cnt["cnt"]], -1, 255, -1)
        self.prev_centroids[f"{color}_helmet"] = helmet_cnt["centroid"]
    else:
        self.prev_centroids[f"{color}_helmet"] = None

    if foot_cnt is not None:
        cv2.drawContours(foot_mask, [foot_cnt["cnt"]], -1, 255, -1)
        self.prev_centroids[f"{color}_foot"] = foot_cnt["centroid"]
    else:
        self.prev_centroids[f"{color}_foot"] = None

    return helmet_mask, foot_mask

# Apply the patch to the class
import types
EquipmentSegmentor._separate_helmet_foot = robust_separate_helmet_foot

def run_evaluation():
    evaluator = PerformanceEvaluator()
    datasets = [
        ("Ghost_Hit", GHOST_HIT_DIR),
        ("Real_Hit", REAL_HIT_DIR),
    ]
    
    augmented_keywords = ["blur", "bright", "dark", "flipped", "noisy"]
    
    for label, directory in datasets:
        if os.path.exists(directory):
            files = sorted([
                f for f in os.listdir(directory)
                if f.endswith(".mp4") and not any(k in f for k in augmented_keywords)
            ])
            
            for vf in files:
                video_path = os.path.join(directory, vf)
                video_name = os.path.splitext(vf)[0]
                
                try:
                    result = analyze_video(video_path, verbose=False)
                    decision = result["decision_result"]["decision"]
                    confidence = result["decision_result"]["confidence"]
                    
                    evaluator.add_result(
                        video_name=video_name,
                        true_label=label,
                        predicted_label=decision,
                        confidence=confidence,
                        details={
                            "contact_frames": result["contact_summary"].get("contact_frames", 0),
                            "max_overlap": result["contact_summary"].get("max_overlap", 0),
                            "min_distance": result["contact_summary"].get("min_distance", float("inf")),
                            "max_acceleration": result["kinematic_result"].get("max_acceleration", 0),
                            "motion_type": result["kinematic_result"].get("type", "N/A"),
                        }
                    )
                    print(f"  {video_name:10s} | True: {label:10s} | Pred: {decision:10s} | Min Dist: {result['contact_summary'].get('min_distance', float('inf')):.1f}px")
                except Exception as e:
                    print(f"  {video_name:10s} | Error: {e}")
                    
    print("\n" + "="*50)
    print("  PERFORMANCE METRICS (ROBUST INIT)")
    print("="*50)
    print(evaluator.print_summary())

if __name__ == "__main__":
    run_evaluation()
