"""Crossover segmentor v3 hızlı test — yalnızca 20 orijinal video."""
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.evaluator import PerformanceEvaluator
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR
from main import analyze_video

def run():
    evaluator = PerformanceEvaluator()
    datasets = [
        ("Ghost_Hit", GHOST_HIT_DIR),
        ("Real_Hit", REAL_HIT_DIR),
    ]
    augmented = ["blur", "bright", "dark", "flipped", "noisy"]

    for label, directory in datasets:
        if not os.path.exists(directory):
            continue
        files = sorted([
            f for f in os.listdir(directory)
            if f.endswith(".mp4") and not any(k in f for k in augmented)
        ])
        for vf in files:
            video_path = os.path.join(directory, vf)
            video_name = os.path.splitext(vf)[0]
            try:
                result = analyze_video(video_path, verbose=False)
                decision = result["decision_result"]["decision"]
                confidence = result["decision_result"]["confidence"]
                min_dist = result["contact_summary"].get("min_distance", float("inf"))
                max_overlap = result["contact_summary"].get("max_overlap", 0)
                motion = result["kinematic_result"].get("type", "N/A")

                evaluator.add_result(
                    video_name=video_name,
                    true_label=label,
                    predicted_label=decision,
                    confidence=confidence,
                    details={
                        "min_distance": min_dist,
                        "max_overlap": max_overlap,
                        "motion_type": motion,
                    }
                )
                match = "OK" if (
                    ("ghost" in label.lower() and ("ghost" in decision.lower() or "external" in decision.lower()))
                    or
                    ("real" in label.lower() and ("real" in decision.lower() or "light" in decision.lower()))
                ) else "XX"
                print(f"  {match} {video_name:10s} | True: {label:10s} | Pred: {decision:18s} | MinDist: {min_dist:7.1f}px | Overlap: {max_overlap:5d} | Motion: {motion}")
            except Exception as e:
                print(f"  !! {video_name:10s} | Error: {e}")

    print("\n" + "=" * 60)
    print(evaluator.print_summary())

if __name__ == "__main__":
    run()
