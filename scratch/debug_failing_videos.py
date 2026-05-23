"""
Her video icin ilk 5 karedeki kontur dagılımını ve
segmentor kararlarnı inceleyen detaylı debug scripti.
Sadece basarısız videolara odaklanır.
"""
import cv2
import numpy as np
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA

def analyze_video_contours(video_path, video_name, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    
    print(f"\n{'='*70}")
    print(f"  VIDEO: {video_name}")
    print(f"{'='*70}")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = preprocessor.process_frame(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        h, w = processed.shape[:2]
        
        # Red contours
        red_mask = segmentor._create_red_mask(hsv)
        red_mask = segmentor._apply_morphology(red_mask)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Blue contours
        blue_mask = segmentor._create_blue_mask(hsv)
        blue_mask = segmentor._apply_morphology(blue_mask)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\n  Frame {frame_idx} (w={w}, h={h})")
        
        for color_name, contours_list in [("RED", red_contours), ("BLUE", blue_contours)]:
            valid = []
            for cnt in contours_list:
                area = cv2.contourArea(cnt)
                if area < 300:  # Show even small ones for debug
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Shape metrics
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * 3.14159 * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                side = "LEFT" if cx < w // 2 else "RIGHT"
                zone = "TOP" if cy < h * 0.4 else ("MID" if cy < h * 0.7 else "BOT")
                
                in_area_range = MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA
                marker = "  " if in_area_range else "XX"
                
                print(f"    {marker} {color_name:4s} | ({cx:4d},{cy:4d}) {side:5s} {zone:3s} | "
                      f"area={area:8.0f} | solid={solidity:.2f} circ={circularity:.2f} | "
                      f"bbox=({bw:3d}x{bh:3d})")
        
        # Run segmentor
        seg = segmentor.segment_frame(processed)
        print(f"    SEGMENTOR OUTPUT:")
        for key in ["red_helmet", "red_foot", "blue_helmet", "blue_foot"]:
            d = seg.get(key, {})
            c = d.get("centroids", [])
            a = d.get("areas", [])
            if c:
                print(f"      {key:15s}: ({c[0][0]:4d},{c[0][1]:4d}) area={a[0]:8.0f}")
            else:
                print(f"      {key:15s}: None")
    
    cap.release()

def main():
    # Focus on failing videos
    failing_videos = [
        ("Ghost_Hit", GHOST_HIT_DIR, ["GH_01", "GH_02", "GH_07"]),
        ("Real_Hit", REAL_HIT_DIR, ["RH_01", "RH_02", "RH_05"]),
    ]
    
    for label, directory, names in failing_videos:
        for name in names:
            vpath = os.path.join(directory, f"{name}.mp4")
            if os.path.exists(vpath):
                analyze_video_contours(vpath, f"{label}/{name}", max_frames=3)

if __name__ == "__main__":
    main()
