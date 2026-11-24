import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.config import HSV_RANGES

def analyze():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    
    # Read a few frames
    for i in range(15):
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = preprocessor.process_frame(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        # Current blue range
        lower = HSV_RANGES["blue_helmet_lower"]
        upper = HSV_RANGES["blue_helmet_upper"]
        
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"--- Frame {i} ---")
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 100:
                # Get HSV values in this contour
                c_mask = np.zeros_like(mask)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                pixels = hsv[c_mask == 255]
                
                h_mean, s_mean, v_mean = np.mean(pixels, axis=0)
                h_min, s_min, v_min = np.min(pixels, axis=0)
                h_max, s_max, v_max = np.max(pixels, axis=0)
                
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
                cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
                
                print(f"Contour {idx:2d} | Area: {area:7.0f} | Center: ({cx},{cy})")
                print(f"  H: {h_min:.0f}-{h_max:.0f} (avg {h_mean:.1f})")
                print(f"  S: {s_min:.0f}-{s_max:.0f} (avg {s_mean:.1f})")
                print(f"  V: {v_min:.0f}-{v_max:.0f} (avg {v_mean:.1f})")
                
    cap.release()

if __name__ == "__main__":
    analyze()
