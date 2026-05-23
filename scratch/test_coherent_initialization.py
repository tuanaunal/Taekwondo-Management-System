import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor

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

        # Optimized border filters
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
                    # Score favors larger area and smaller horizontal distance
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
            # Look for a foot relative to this helmet
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

def test():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    
    # Bind patch
    import types
    segmentor._separate_helmet_foot = types.MethodType(robust_separate_helmet_foot, segmentor)
    
    frame_idx = 0
    print("Frame | Red Helmet | Red Foot | Blue Helmet | Blue Foot")
    print("-" * 65)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = preprocessor.process_frame(frame)
        seg_results = segmentor.segment_frame(processed)
        
        rh = seg_results["red_helmet"]["centroids"]
        rf = seg_results["red_foot"]["centroids"]
        bh = seg_results["blue_helmet"]["centroids"]
        bf = seg_results["blue_foot"]["centroids"]
        
        rh_str = str(rh[0]) if rh else "None"
        rf_str = str(rf[0]) if rf else "None"
        bh_str = str(bh[0]) if bh else "None"
        bf_str = str(bf[0]) if bf else "None"
        
        print(f"{frame_idx:5d} | {rh_str:10s} | {rf_str:8s} | {bh_str:11s} | {bf_str:9s}")
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    test()
