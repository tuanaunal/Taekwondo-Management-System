import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.contact_analyzer import ContactAnalyzer

def find():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    contact_analyzer = ContactAnalyzer()
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = preprocessor.process_frame(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        red_mask = segmentor._create_red_mask(hsv)
        red_mask = segmentor._apply_morphology(red_mask)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_mask = segmentor._create_blue_mask(hsv)
        blue_mask = segmentor._apply_morphology(blue_mask)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter raw contours by area to ignore very small noise
        valid_red = [c for c in red_contours if cv2.contourArea(c) > 300]
        valid_blue = [c for c in blue_contours if cv2.contourArea(c) > 300]
        
        # Check all pairs
        for r_idx, r_cnt in enumerate(valid_red):
            r_area = cv2.contourArea(r_cnt)
            r_M = cv2.moments(r_cnt)
            if r_M["m00"] == 0: continue
            r_cx = int(r_M["m10"] / r_M["m00"])
            r_cy = int(r_M["m01"] / r_M["m00"])
            
            for b_idx, b_cnt in enumerate(valid_blue):
                b_area = cv2.contourArea(b_cnt)
                b_M = cv2.moments(b_cnt)
                if b_M["m00"] == 0: continue
                b_cx = int(b_M["m10"] / b_M["m00"])
                b_cy = int(b_M["m01"] / b_M["m00"])
                
                # Check pixel overlap
                overlap_pixels = 0
                r_mask_cnt = np.zeros_like(red_mask)
                cv2.drawContours(r_mask_cnt, [r_cnt], -1, 255, -1)
                
                b_mask_cnt = np.zeros_like(blue_mask)
                cv2.drawContours(b_mask_cnt, [b_cnt], -1, 255, -1)
                
                overlap = cv2.bitwise_and(r_mask_cnt, b_mask_cnt)
                overlap_pixels = cv2.countNonZero(overlap)
                
                dist = contact_analyzer._compute_min_contour_distance([r_cnt], [b_cnt])
                
                if dist < 120 or overlap_pixels > 0:
                    print(f"Frame {frame_idx:2d} | Red({r_cx},{r_cy},area={r_area:.0f}) to Blue({b_cx},{b_cy},area={b_area:.0f}) | Dist: {dist:.1f}px | Overlap: {overlap_pixels}px")
                    
        frame_idx += 1
    cap.release()

if __name__ == "__main__":
    find()
