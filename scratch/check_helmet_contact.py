import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from main import analyze_video

def check():
    # Let's run RH_01.mp4 and inspect the frame-by-frame details of all segmented items
    video_path = os.path.join(project_root, "Dataset", "Real_Hit", "RH_01.mp4")
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    
    # Use class patched separate_helmet_foot
    import scratch.test_all_patched
    EquipmentSegmentor._separate_helmet_foot = scratch.test_all_patched.patched_separate_helmet_foot
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = preprocessor.process_frame(frame)
        seg_results = segmentor.segment_frame(processed)
        
        # Check distance between ALL pairs
        print(f"Frame {frame_idx}:")
        for k1 in ["red_helmet", "red_foot", "blue_helmet", "blue_foot"]:
            for k2 in ["red_helmet", "red_foot", "blue_helmet", "blue_foot"]:
                if k1 >= k2 or k1.split("_")[0] == k2.split("_")[0]:
                    continue
                c1 = seg_results[k1]["centroids"]
                c2 = seg_results[k2]["centroids"]
                if c1 and c2:
                    dist = np.linalg.norm(np.array(c1[0]) - np.array(c2[0]))
                    print(f"  {k1} - {k2}: {dist:.1f}px (c1: {c1[0]}, c2: {c2[0]})")
        frame_idx += 1
    cap.release()

if __name__ == "__main__":
    check()
