import cv2
import numpy as np
import os
import sys

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor

def inspect():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    
    # Patched separate_helmet_foot with optimized border margin filtering
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
            contours_with_dists = []
            for c in valid_contours:
                d_h = np.linalg.norm(np.array(c["centroid"]) - np.array(prev_h))
                d_f = np.linalg.norm(np.array(c["centroid"]) - np.array(prev_f))
                contours_with_dists.append({
                    "data": c,
                    "dist_h": d_h,
                    "dist_f": d_f
                })

            contours_with_dists.sort(key=lambda x: x["dist_h"])
            helmet_match = contours_with_dists[0]
            if helmet_match["dist_h"] < 350:
                helmet_cnt = helmet_match["data"]

            remaining = [c for c in contours_with_dists if c["data"] is not helmet_cnt]
            if remaining:
                remaining.sort(key=lambda x: x["dist_f"])
                foot_match = remaining[0]
                
                is_hogu = False
                if helmet_cnt is not None:
                    y_diff = foot_match["data"]["centroid"][1] - helmet_cnt["centroid"][1]
                    area_ratio = foot_match["data"]["area"] / (helmet_cnt["area"] + 1e-5)
                    if 0 < y_diff < 220 and area_ratio > 1.2:
                        is_hogu = True
                
                if foot_match["dist_f"] < 350 and not is_hogu:
                    foot_cnt = foot_match["data"]
        else:
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

    # Bind the patched method
    import types
    segmentor._separate_helmet_foot = types.MethodType(patched_separate_helmet_foot, segmentor)

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
    inspect()
