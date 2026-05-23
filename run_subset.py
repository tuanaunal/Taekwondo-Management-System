"""
run_subset.py — Orijinal 20 Video Üzerinde Hızlı Performans Değerlendirmesi
========================================================================
Artırılmamış orijinal 20 videoyu analiz eder ve hızlı sonuç raporu sunar.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR
from src.evaluator import PerformanceEvaluator
from main import analyze_video

def run_subset_analysis():
    evaluator = PerformanceEvaluator()
    all_results = []

    datasets = [
        ("Ghost_Hit", GHOST_HIT_DIR, "GHOST_HIT"),
        ("Real_Hit", REAL_HIT_DIR, "REAL_HIT"),
    ]

    # Sadece orijinal videoları seç (isimlerinde augment etiketleri olmayanlar)
    augmented_keywords = ["blur", "bright", "dark", "flipped", "noisy"]

    selected_videos = {}
    total_videos = 0

    for label, directory, _ in datasets:
        if os.path.exists(directory):
            files = sorted([
                f for f in os.listdir(directory)
                if f.endswith(".mp4") and not any(k in f for k in augmented_keywords)
            ])
            selected_videos[label] = (directory, files)
            total_videos += len(files)

    print("=" * 60)
    print("  TAEKWONDO GHOST HIT — ORIJINAL 20 VIDEO ANALIZI")
    print("=" * 60)
    print(f"  Toplam seçilen orijinal video: {total_videos}")
    print()

    current = 0

    for label, (directory, video_files) in selected_videos.items():
        print(f"\n{'─' * 50}")
        print(f"  Kategori: {label} ({len(video_files)} video)")
        print(f"{'─' * 50}")

        for vf in video_files:
            current += 1
            video_path = os.path.join(directory, vf)
            video_name = os.path.splitext(vf)[0]

            print(f"\n  [{current}/{total_videos}] {vf} analiz ediliyor...")

            try:
                result = analyze_video(video_path, verbose=False)

                decision = result["decision_result"]["decision"]
                confidence = result["decision_result"]["confidence"]
                indicator = result["decision_result"]["indicator"]
                label_tr = result["decision_result"]["label_tr"]

                evaluator.add_result(
                    video_name=video_name,
                    true_label=label,
                    predicted_label=decision,
                    confidence=confidence,
                    details={
                        "contact_frames": result["contact_summary"].get("contact_frames", 0),
                        "max_overlap": result["contact_summary"].get("max_overlap", 0),
                        "min_distance": result["contact_summary"].get("min_distance", float("inf")),
                        "max_acceleration": result["kinematic_result"].get("max_acceleration", 0),
                        "motion_type": result["kinematic_result"].get("type", "N/A"),
                    },
                )

                all_results.append({
                    "video": video_name,
                    "true_label": label,
                    "predicted": decision,
                    "label_tr": label_tr,
                    "confidence": confidence,
                })

                print(f"    {indicator} {label_tr} (güven: {confidence:.1%})")

            except Exception as e:
                print(f"    HATA: {e}")
                evaluator.add_result(
                    video_name=video_name,
                    true_label=label,
                    predicted_label="INCONCLUSIVE",
                    confidence=0.0,
                    details={"error": str(e)},
                )

    print("\n\n" + "=" * 60)
    print("  RAPOR VE METRIKLER (ORIJINAL 20 VIDEO)")
    print("=" * 60)
    summary = evaluator.print_summary()
    print(summary)

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "─" * 60)
        print("  VIDEO BAZLI SONUÇLAR")
        print("─" * 60)
        for _, row in df.iterrows():
            match_icon = "✓" if (
                ("ghost" in row["true_label"].lower() and "ghost" in row["predicted"].lower()) or
                ("real" in row["true_label"].lower() and "real" in row["predicted"].lower())
            ) else "✗"
            print(f"  {match_icon} {row['video']:12s} | Gerçek: {row['true_label']:10s} | "
                  f"Tahmin: {row['label_tr']:30s} | Güven: {row['confidence']:.1%}")

if __name__ == "__main__":
    run_subset_analysis()
