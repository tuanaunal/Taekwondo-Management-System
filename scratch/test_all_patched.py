import cv2
import numpy as np
import os
import sys

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.evaluator import PerformanceEvaluator
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR
from main import analyze_video

# Patched separate_helmet_foot with optimized bipartite matching and border filtering
def patched_separate_helmet_foot(self, mask: np.ndarray, frame: np.ndarray, color: str) -> tuple:
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

        # Optimized border noise filters:
        # - Left border (cx < 180)
        # - Right border (cx > w - 180)
        # - Top border / Scoreboard / Ceiling lights (cy < 130)
        # - Bottom border / Floor lines / Static ground noise (cy > h - 140)
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

    if prev_h is not None and prev_f is not None:
        # Bipartite matching for 2 objects to minimize total distance
        best_score = float("inf")
        
        if len(valid_contours) >= 2:
            for i in range(len(valid_contours)):
                for j in range(len(valid_contours)):
                    if i == j:
                        continue
                    c_h = valid_contours[i]
                    c_f = valid_contours[j]
                    
                    # Hogu filter
                    y_diff = c_f["centroid"][1] - c_h["centroid"][1]
                    area_ratio = c_f["area"] / (c_h["area"] + 1e-5)
                    if 0 < y_diff < 220 and area_ratio > 1.2:
                        continue
                        
                    d_h = np.linalg.norm(np.array(c_h["centroid"]) - np.array(prev_h))
                    d_f = np.linalg.norm(np.array(c_f["centroid"]) - np.array(prev_f))
                    total_dist = d_h + d_f
                    
                    if total_dist < best_score:
                        best_score = total_dist
                        helmet_cnt = c_h
                        foot_cnt = c_f
            
            # If tracking was completely lost, reset
            if best_score > 600:
                helmet_cnt = None
                foot_cnt = None
                
        elif len(valid_contours) == 1:
            # Match single contour to closest role
            c = valid_contours[0]
            d_h = np.linalg.norm(np.array(c["centroid"]) - np.array(prev_h))
            d_f = np.linalg.norm(np.array(c["centroid"]) - np.array(prev_f))
            
            if d_h < d_f and d_h < 350:
                helmet_cnt = c
            elif d_f < d_h and d_f < 350:
                is_hogu = False
                if prev_h is not None:
                    y_diff = c["centroid"][1] - prev_h[1]
                    if 0 < y_diff < 220:
                        is_hogu = True
                if not is_hogu:
                    foot_cnt = c
                    
    else:
        # Fallback to position-based initialization
        valid_contours.sort(key=lambda x: x["centroid"][1])
        if len(valid_contours) >= 2:
            helmet_cnt = valid_contours[0]
            helmet_cy = helmet_cnt["centroid"][1]
            foot_candidates = [
                c for c in valid_contours[1:]
                if (c["centroid"][1] - helmet_cy) >= 250
            ]
            if foot_candidates:
                foot_cnt = foot_candidates[-1]
        elif len(valid_contours) == 1:
            cnt1 = valid_contours[0]
            if cnt1["centroid"][1] < mid_y:
                helmet_cnt = cnt1
            else:
                helmet_cnt = cnt1

    if helmet_cnt is not None:
        cv2.drawContours(helmet_mask, [helmet_cnt["cnt"]], -1, 255, -1)
        self.prev_centroids[f"{color}_helmet"] = helmet_cnt["centroid"]

    if foot_cnt is not None:
        cv2.drawContours(foot_mask, [foot_cnt["cnt"]], -1, 255, -1)
        self.prev_centroids[f"{color}_foot"] = foot_cnt["centroid"]

    return helmet_mask, foot_mask

# Bind patch to EquipmentSegmentor class level
EquipmentSegmentor._separate_helmet_foot = patched_separate_helmet_foot

def run_all_subset():
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
    print("  PERFORMANCE METRICS")
    print("="*50)
    print(evaluator.print_summary())

if __name__ == "__main__":
    run_all_subset()
