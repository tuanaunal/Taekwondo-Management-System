"""
Başarısız Real Hit videoları için kare-kare detaylı debug.
Segmentörün ayak ve kaskı nereye koyduğunu, temas anındaki 
mesafeleri ve kaçırılan konturları analiz eder.
"""
import cv2
import numpy as np
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.contact_analyzer import ContactAnalyzer
from src.config import REAL_HIT_DIR

def debug_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    contact_analyzer = ContactAnalyzer()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*70}")
    print(f"  {video_name} ({total_frames} frames)")
    print(f"{'='*70}")
    
    min_dist_all = float("inf")
    best_frame = -1
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = preprocessor.process_frame(frame)
        seg = segmentor.segment_frame(processed)
        
        # Her kare çifti için temas hesapla
        for foot_key, helmet_key in [("red_foot", "blue_helmet"), ("blue_foot", "red_helmet")]:
            contact = contact_analyzer.analyze_contact(
                seg.get(helmet_key, {}),
                seg.get(foot_key, {}),
            )
            dist = contact.get("min_contour_distance", float("inf"))
            overlap = contact.get("overlap_pixels", 0)
            
            f_c = seg.get(foot_key, {}).get("centroids", [])
            h_c = seg.get(helmet_key, {}).get("centroids", [])
            f_a = seg.get(foot_key, {}).get("areas", [])
            h_a = seg.get(helmet_key, {}).get("areas", [])
            
            if dist < min_dist_all:
                min_dist_all = dist
                best_frame = frame_idx
            
            # Sadece ilginç kareler (temas yakın veya her 10 karede bir)
            if dist < 60 or overlap > 0 or frame_idx % 10 == 0:
                f_str = f"({f_c[0][0]:4d},{f_c[0][1]:4d}) a={f_a[0]:6.0f}" if f_c else "None"
                h_str = f"({h_c[0][0]:4d},{h_c[0][1]:4d}) a={h_a[0]:6.0f}" if h_c else "None"
                
                print(f"  F{frame_idx:3d} | {foot_key:10s}->{helmet_key:12s} | "
                      f"dist={dist:6.1f} olap={overlap:5d} | "
                      f"foot={f_str} helm={h_str}")
        
        frame_idx += 1
    
    cap.release()
    print(f"  --- Best contact: frame {best_frame}, min_dist={min_dist_all:.1f}px")

def main():
    failing_rh = ["RH_02", "RH_03", "RH_04", "RH_05", "RH_07", "RH_08", "RH_10"]
    for name in failing_rh:
        vpath = os.path.join(REAL_HIT_DIR, f"{name}.mp4")
        if os.path.exists(vpath):
            debug_video(vpath, name)

if __name__ == "__main__":
    main()
