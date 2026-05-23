import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.config import MIN_CONTOUR_AREA, MAX_CONTOUR_AREA
from main import analyze_video

def patched_segment_frame(self, frame: np.ndarray, fg_mask: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = {}

    # Apply fg_mask to frame/hsv or to masks to remove static background
    # ── Kırmızı Ekipman (Kask + Ayak) ──
    red_mask = self._create_red_mask(hsv)
    red_mask = cv2.bitwise_and(red_mask, fg_mask) # Filter with foreground
    red_mask = self._apply_morphology(red_mask)
    red_helmet, red_foot = self._separate_helmet_foot(red_mask, frame, "red")

    results["red_helmet"] = self._extract_features(red_helmet)
    results["red_foot"] = self._extract_features(red_foot)

    # ── Mavi Ekipman (Kask + Ayak) ──
    blue_mask = self._create_blue_mask(hsv)
    blue_mask = cv2.bitwise_and(blue_mask, fg_mask) # Filter with foreground
    blue_mask = self._apply_morphology(blue_mask)
    blue_helmet, blue_foot = self._separate_helmet_foot(blue_mask, frame, "blue")

    results["blue_helmet"] = self._extract_features(blue_helmet)
    results["blue_foot"] = self._extract_features(blue_foot)

    return results

def test():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    
    # Apply patches
    import scratch.test_all_patched
    EquipmentSegmentor._separate_helmet_foot = scratch.test_all_patched.patched_separate_helmet_foot
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = preprocessor.process_frame(frame)
        
        # MOG foreground mask
        fg_mask = preprocessor.bg_subtractor.apply(processed)
        # Clean up fg_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        seg_results = patched_segment_frame(segmentor, processed, fg_mask)
        
        rh = seg_results["red_helmet"]["centroids"]
        rf = seg_results["red_foot"]["centroids"]
        bh = seg_results["blue_helmet"]["centroids"]
        bf = seg_results["blue_foot"]["centroids"]
        
        rh_str = str(rh[0]) if rh else "None"
        rf_str = str(rf[0]) if rf else "None"
        bh_str = str(bh[0]) if bh else "None"
        bf_str = str(bf[0]) if bf else "None"
        
        print(f"{frame_idx:2d} | RH: {rh_str:10s} | RF: {rf_str:8s} | BH: {bh_str:11s} | BF: {bf_str:9s}")
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    test()
