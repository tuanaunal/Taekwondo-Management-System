import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.config import MIN_CONTOUR_AREA, MAX_CONTOUR_AREA

def inspect_frame_31():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    
    # Use class patched separate_helmet_foot
    import scratch.test_all_patched
    EquipmentSegmentor._separate_helmet_foot = scratch.test_all_patched.patched_separate_helmet_foot
    
    for i in range(32): # Up to frame 31
        ret, frame = cap.read()
        
    processed = preprocessor.process_frame(frame)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    
    # Create blue mask and apply morphology
    blue_mask = segmentor._create_blue_mask(hsv)
    blue_mask = segmentor._apply_morphology(blue_mask)
    
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"--- Frame 31 Contours ---")
    print(f"Total contours: {len(contours)}")
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
        
        # Border check
        h, w = blue_mask.shape[:2]
        is_border = cx < 180 or cx > w - 180 or cy < 130 or cy > h - 140
        
        print(f"Contour {idx} | Area: {area:.0f} | Center: ({cx},{cy}) | Border: {is_border} | Min/Max Area Valid: {MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA}")
        
    # Let's see what _separate_helmet_foot did in frame 31
    helmet_mask, foot_mask = segmentor._separate_helmet_foot(blue_mask, processed, "blue")
    
    # Check what was segmented
    hc, _ = cv2.findContours(helmet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fc, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("\n--- Segmentor Decisions ---")
    print(f"Helmet Count: {len(hc)}")
    if hc:
        ha = cv2.contourArea(hc[0])
        hM = cv2.moments(hc[0])
        hcx = int(hM["m10"] / hM["m00"]) if hM["m00"] > 0 else 0
        hcy = int(hM["m01"] / hM["m00"]) if hM["m00"] > 0 else 0
        print(f"  Helmet Center: ({hcx},{hcy}) | Area: {ha}")
        
    print(f"Foot Count: {len(fc)}")
    if fc:
        fa = cv2.contourArea(fc[0])
        fM = cv2.moments(fc[0])
        fcx = int(fM["m10"] / fM["m00"]) if fM["m00"] > 0 else 0
        fcy = int(fM["m01"] / fM["m00"]) if fM["m00"] > 0 else 0
        print(f"  Foot Center: ({fcx},{fcy}) | Area: {fa}")
        
    cap.release()

if __name__ == "__main__":
    inspect_frame_31()
