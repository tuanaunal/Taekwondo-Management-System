"""
run_batch.py — Toplu Video Analizi ve Performans Değerlendirmesi
=================================================================
Dataset klasöründeki tüm videoları otomatik olarak analiz eder,
sonuçları karşılaştırır ve performans metriklerini raporlar.

Kullanım:
    python run_batch.py
"""

import sys
import os

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR, REPORTS_DIR, GRAPHS_DIR, LOGS_DIR
from src.evaluator import PerformanceEvaluator
from main import analyze_video


def run_batch_analysis():
    """
    Tüm Ghost_Hit ve Real_Hit videolarını analiz eder
    ve performans değerlendirmesi yapar.
    """
    evaluator = PerformanceEvaluator()
    all_results = []

    # ── Veri seti tanımları ──
    datasets = [
        ("Ghost_Hit", GHOST_HIT_DIR, "GHOST_HIT"),
        ("Real_Hit", REAL_HIT_DIR, "REAL_HIT"),
    ]

    total_videos = 0
    for label, directory, _ in datasets:
        if os.path.exists(directory):
            total_videos += len([f for f in os.listdir(directory) if f.endswith(".mp4")])

    print("=" * 60)
    print("  TAEKWONDO GHOST HIT DETECTION — TOPLU ANALİZ")
    print("=" * 60)
    print(f"  Toplam video: {total_videos}")
    print()

    current = 0

    for label, directory, true_decision in datasets:
        if not os.path.exists(directory):
            print(f"  UYARI: Dizin bulunamadı: {directory}")
            continue

        video_files = sorted([f for f in os.listdir(directory) if f.endswith(".mp4")])
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

                # Evaluator'a kaydet
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

                net_disp = result["kinematic_result"].get("net_displacement", 0)
                max_acc = result["kinematic_result"].get("max_acceleration", 0)
                kinetic_energy = (net_disp * 1.5) + (max_acc * 0.5)
                min_distance = result["contact_summary"].get("min_distance", float("inf"))
                max_overlap = result["contact_summary"].get("max_overlap", 0)

                all_results.append({
                    "video": video_name,
                    "true_label": label,
                    "predicted": decision,
                    "label_tr": label_tr,
                    "confidence": confidence,
                    "graph_path": result.get("graph_path", ""),
                    "net_disp": net_disp,
                    "max_acc": max_acc,
                    "kinetic_energy": kinetic_energy,
                    "min_distance": min_distance,
                    "max_overlap": max_overlap
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

    # ── Performans Değerlendirmesi ──
    print("\n\n")
    summary = evaluator.print_summary()
    print(summary)

    # ── Grafikler ──
    cm_path = evaluator.plot_confusion_matrix()
    if cm_path:
        print(f"\n  Confusion Matrix kaydedildi: {cm_path}")

    metrics_path = evaluator.plot_metrics_summary()
    if metrics_path:
        print(f"  Metrik özeti kaydedildi: {metrics_path}")

    # ── CSV Rapor ──
    report_path = evaluator.generate_report()
    if report_path:
        print(f"  Detaylı rapor kaydedildi: {report_path}")

    # ── Sonuç özet tablosu ──
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(REPORTS_DIR, "batch_summary.csv")
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"  Özet tablo kaydedildi: {summary_path}")

        # Konsola özet tablo
        print("\n" + "─" * 60)
        print("  VIDEO BAZLI SONUÇLAR")
        print("─" * 60)
        for row in all_results:
            is_true_ghost = "ghost" in row["true_label"].lower()
            is_pred_ghost = "ghost" in row["predicted"].lower() or "external" in row["predicted"].lower()
            match_icon = "✓" if (is_true_ghost == is_pred_ghost) else "✗"
            print(f"  {match_icon} {row['video']:12s} | Gerçek: {row['true_label']:10s} | "
                  f"Tahmin: {row['label_tr']:30s} | Güven: {row['confidence']:.1%}")

    # Sadece CSV formatında raporlama
    csv_path = os.path.join(REPORTS_DIR, "batch_summary_detailed.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("video,true_label,predicted,label_tr,confidence,net_disp,max_acc,kinetic_energy,min_distance,max_overlap,graph_path\n")
        for r in all_results:
            f.write(f"{r['video']},{r['true_label']},{r['predicted']},{r['label_tr']},{r['confidence']},{r['net_disp']},{r['max_acc']},{r['kinetic_energy']},{r['min_distance']},{r['max_overlap']},{r['graph_path']}\n")

    print("\n" + "="*60)
    print("  Toplu analiz tamamlandı!")
    print("="*60 + "\n")

    return evaluator.compute_metrics()


if __name__ == "__main__":
    run_batch_analysis()
