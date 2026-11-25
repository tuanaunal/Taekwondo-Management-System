import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor

def find_range():
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    
    # Read frame 10 where the blue mat is very prominent
    for _ in range(11):
        ret, frame = cap.read()
    
    processed = preprocessor.process_frame(frame)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    
    # Let's segment with a very broad blue mask first to find the giant mat contour
    broad_lower = np.array([90, 50, 40])
    broad_upper = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, broad_lower, broad_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (this is the blue mat)
    mat_cnt = max(contours, key=cv2.contourArea)
    print(f"Blue mat contour area: {cv2.contourArea(mat_cnt)}")
    
    # Extract HSV values of the mat
    mat_mask = np.zeros_like(mask)
    cv2.drawContours(mat_mask, [mat_cnt], -1, 255, -1)
    mat_pixels = hsv[mat_mask == 255]
    
    h = mat_pixels[:, 0]
    s = mat_pixels[:, 1]
    v = mat_pixels[:, 2]
    
    print("Blue Mat HSV percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}% | H: {np.percentile(h, p):.1f} | S: {np.percentile(s, p):.1f} | V: {np.percentile(v, p):.1f}")
        
    # Now let's find the blue helmet/foot in this frame.
    # We know the blue helmet is usually around the upper-middle of the screen, and blue foot is also smaller.
    print("\nSmaller blue contours (likely equipment):")
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if 1000 < area < 50000:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            
            c_mask = np.zeros_like(mask)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            eq_pixels = hsv[c_mask == 255]
            eh = eq_pixels[:, 0]
            es = eq_pixels[:, 1]
            ev = eq_pixels[:, 2]
            
            print(f"Contour {idx} | Area: {area:.0f} | Center: ({cx},{cy})")
            print(f"  H: {np.percentile(eh, 5):.1f} - {np.percentile(eh, 95):.1f}")
            print(f"  S: {np.percentile(es, 5):.1f} - {np.percentile(es, 95):.1f}")
            print(f"  V: {np.percentile(ev, 5):.1f} - {np.percentile(ev, 95):.1f}")
            
    cap.release()

if __name__ == "__main__":
    find_range()
