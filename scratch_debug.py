import sys
import os

# Proje kok dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import analyze_video

def main():
    video_path = "Dataset/Real_Hit/RH_01.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    res = analyze_video(video_path, verbose=False)
    log = res["frame_log"]
    
    print("Frames where contour_dist < 150 or overlap > 0:")
    for f in log:
        if f["contour_dist"] < 150 or f["overlap_px"] > 0:
            print(f"Frame {f['frame']:02d}: red_foot=({f.get('red_foot_x')}, {f.get('red_foot_y')}), "
                  f"blue_helmet=({f.get('blue_helmet_x')}, {f.get('blue_helmet_y')}), "
                  f"dist={f['contour_dist']:.2f}, overlap={f['overlap_px']}")

if __name__ == "__main__":
    main()
