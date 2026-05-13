"""
main.py — Taekwondo Ghost Hit Detection: CLI Giris Noktasi
============================================================
Tek bir video dosyasini analiz ederek Ghost Hit / Real Hit karari verir.

Kullanim:
    python main.py <video_yolu>
    python main.py Dataset/Ghost_Hit/GH_01.mp4
"""

import sys
import os
import io
import cv2
import numpy as np

# Windows konsol encoding duzeltmesi
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Proje kok dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import GRAPHS_DIR, LOGS_DIR, TARGET_FPS
from src.preprocessor import VideoPreprocessor, load_video, get_video_info
from src.segmentor import EquipmentSegmentor
from src.tracker import MultiObjectTracker
from src.kinematics import KinematicAnalyzer
from src.contact_analyzer import ContactAnalyzer
from src.decision_engine import DecisionEngine


def analyze_video(video_path: str, verbose: bool = True) -> dict:
    """
    Tek bir video dosyasini uctan uca analiz eder.

    Parameters
    ----------
    video_path : str
        Analiz edilecek video dosya yolu.
    verbose : bool
        Ayrintili cikti yazdir.

    Returns
    -------
    dict
        {
            'decision_result': dict,    # DecisionEngine karari
            'contact_summary': dict,    # Temas ozeti
            'kinematic_result': dict,   # Kinematik sonuc
            'frame_log': list,          # Kare bazli log
            'graph_path': str,          # Ivme grafigi dosya yolu
            'video_info': dict,         # Video meta bilgileri
        }
    """
    # -- Modulleri baslat --
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    tracker = MultiObjectTracker(max_disappeared=5)
    kinematic_analyzer = KinematicAnalyzer(fps=TARGET_FPS)
    contact_analyzer = ContactAnalyzer()
    decision_engine = DecisionEngine()

    # -- Video ac --
    cap = load_video(video_path)
    info = get_video_info(cap)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"  Video: {video_name}")
        print(f"  Cozunurluk: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']:.0f} | Kare: {info['frame_count']}")
        print(f"  Sure: {info['duration_sec']:.2f}s")
        print(f"{'-' * 50}")

    # -- Kare-kare analiz --
    frame_contacts = []
    frame_log = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. On-isleme
        processed = preprocessor.process_frame(frame)

        # 2. Segmentasyon
        seg_results = segmentor.segment_frame(processed)

        # 3. Takip
        tracking = tracker.update(seg_results, frame_idx)

        # 4. Temas analizi (her renk cifti icin)
        # Kirmizi ayak -> Mavi kask (veya tam tersi)
        contact_pairs = [
            ("red_foot", "blue_helmet"),
            ("blue_foot", "red_helmet"),
        ]

        best_contact = None
        for foot_key, helmet_key in contact_pairs:
            contact = contact_analyzer.analyze_contact(
                seg_results.get(helmet_key, {}),
                seg_results.get(foot_key, {}),
            )
            if best_contact is None or contact["overlap_pixels"] > best_contact["overlap_pixels"]:
                best_contact = contact

        if best_contact is None:
            best_contact = {
                "has_contact": False,
                "contact_type": "no_detection",
                "overlap_pixels": 0,
                "centroid_distance": float("inf"),
                "min_contour_distance": float("inf"),
                "confidence": 0.0,
                "details": "Hicbir ekipman cifti tespit edilemedi.",
            }

        frame_contacts.append(best_contact)

        # Log kaydi
        log_entry = {
            "frame": frame_idx,
            "time_sec": frame_idx / info["fps"],
            "contact_type": best_contact["contact_type"],
            "overlap_px": best_contact["overlap_pixels"],
            "centroid_dist": best_contact["centroid_distance"],
            "contour_dist": best_contact["min_contour_distance"],
        }

        # Takip edilen nesnelerin konumlarini ekle
        for key in ["red_helmet", "blue_helmet", "red_foot", "blue_foot"]:
            centroids = seg_results.get(key, {}).get("centroids", [])
            if centroids:
                log_entry[f"{key}_x"] = centroids[0][0]
                log_entry[f"{key}_y"] = centroids[0][1]
            else:
                log_entry[f"{key}_x"] = None
                log_entry[f"{key}_y"] = None

        frame_log.append(log_entry)
        frame_idx += 1

        if verbose and frame_idx % 10 == 0:
            print(f"  Islenen kare: {frame_idx}/{info['frame_count']}")

    cap.release()

    if verbose:
        print(f"  Toplam islenen kare: {frame_idx}")

    # -- Temas Ozeti --
    contact_summary = contact_analyzer.analyze_video_contacts(frame_contacts)

    # -- Kinematik Analiz --
    # En cok hareket eden kask yorungesini bul
    all_traj = tracker.get_all_trajectories()
    best_trajectory = []
    best_traj_key = None

    for key in ["red_helmet", "blue_helmet"]:
        trajs = all_traj.get(key, {})
        for obj_id, traj in trajs.items():
            if len(traj) > len(best_trajectory):
                best_trajectory = traj
                best_traj_key = key

    kinematic_result = kinematic_analyzer.classify_motion(best_trajectory)

    # Ivme grafigi olustur
    graph_path = ""
    if len(best_trajectory) >= 3:
        graph_path = kinematic_analyzer.plot_kinematics(
            best_trajectory,
            title=f"{video_name} - Kask Kinematigi",
        )
        if verbose:
            print(f"  Grafik kaydedildi: {graph_path}")

    # -- Nihai Karar --
    decision_result = decision_engine.make_decision(
        contact_summary, kinematic_result,
    )

    if verbose:
        report = decision_engine.format_report(decision_result)
        print(report)

    return {
        "decision_result": decision_result,
        "contact_summary": contact_summary,
        "kinematic_result": kinematic_result,
        "frame_log": frame_log,
        "graph_path": graph_path,
        "video_info": info,
    }


def launch_gui():
    """PyQt5 GUI arayuzunu baslatir."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def main():
    """
    Giris noktasi.
    - Arguman verilmezse: GUI arayuzu acilir
    - Arguman verilirse: CLI modunda tek video analiz edilir
    """
    if len(sys.argv) < 2:
        # Arguman yoksa direkt GUI ac
        launch_gui()
        return

    video_path = sys.argv[1]

    # Goreceli yolu mutlak yola cevir
    if not os.path.isabs(video_path):
        video_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            video_path,
        )

    if not os.path.exists(video_path):
        print(f"Hata: Video dosyasi bulunamadi: {video_path}")
        sys.exit(1)

    result = analyze_video(video_path, verbose=True)
    return result


if __name__ == "__main__":
    main()