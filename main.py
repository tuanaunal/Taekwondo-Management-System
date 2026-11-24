"""
main.py — Taekwondo Ghost Hit Detection: Tek Giris Noktasi
============================================================
Projenin tek giris noktasidir.

Kullanim:
    python main.py                               -> Arayuz (GUI) acilir
    python main.py <video_yolu>                  -> Tek video komut satirinda analiz edilir
    python main.py Dataset/Ghost_Hit/GH_01.mp4

Not: Windows'ta PyQt5, PyTorch'tan once yuklenirse torch DLL'leri acilamiyor
(WinError 1114). Bu dosya src modullerini (ve dolayisiyla torch'u) en basta
yukledigi icin arayuz sorunsuz acilir.
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

from src.config import GRAPHS_DIR, LOGS_DIR
from src.preprocessor import VideoPreprocessor, load_video, get_video_info
from src.segmentor import EquipmentSegmentor
from src.tracker import MultiObjectTracker
from src.kinematics import KinematicAnalyzer
from src.contact_analyzer import ContactAnalyzer
from src.decision_engine import DecisionEngine
from src.contact_moment import find_contact_moment, METHOD_USER, METHOD_APPROACH
from src.head_response import compute_head_response


def analyze_video(video_path: str, verbose: bool = True, save_graph: bool = True,
                  contact_frame: int = None) -> dict:
    """
    Tek bir video dosyasini uctan uca analiz eder.

    Parameters
    ----------
    video_path : str
        Analiz edilecek video dosya yolu.
    verbose : bool
        Ayrintili cikti yazdir.
    save_graph : bool
        Ivme grafigini output/graphs klasorune kaydet (toplu testte kapatilir).
    contact_frame : int, optional
        Kullanicinin (orn. hakemin) isaretledigi, sensorun puan verdigi kare.
        Verilmezse temas ani otomatik bulunur.

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
            'contact_moment': dict,     # Temas ani (kare, yontem, hedef sporcu)
            'head_response': dict,      # Bas tepkisi: govdeye gore savrulma ve bas donusu
        }
    """
    # -- Modulleri baslat --
    preprocessor = VideoPreprocessor()
    segmentor = EquipmentSegmentor()
    tracker = MultiObjectTracker(max_disappeared=5)
    contact_analyzer = ContactAnalyzer()
    decision_engine = DecisionEngine()

    # -- Video ac --
    cap = load_video(video_path)
    info = get_video_info(cap)
    # Hız ve ivme, videonun gerçek FPS değeriyle hesaplanır (60 FPS / ağır çekim
    # videolarda 30 FPS varsaymak ivmeleri 4 kata kadar yanlış çıkarır).
    kinematic_analyzer = KinematicAnalyzer(fps=info["fps"])
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"  Video: {video_name}")
        print(f"  Cozunurluk: {info['width']}x{info['height']}")
        fps_note = " (okunamadi, varsayilan)" if info["fps_estimated"] else ""
        print(f"  FPS: {info['fps']:.0f}{fps_note} | Kare: {info['frame_count']}")
        print(f"  Sure: {info['duration_sec']:.2f}s")
        print(f"{'-' * 50}")

    # -- Kare-kare analiz --
    frame_contacts = []
    frame_log = []
    frame_idx = 0

    # 1-2. On-isleme ve sporcu tespiti (iki gecis):
    #   1. gecis: tum karelerde kisiler tespit edilir, kalici takip numaralari ve renk oylari toplanir
    #   2. gecis: kirmizi/mavi sporcu video geneline gore secilir, her kare icin ozellikler uretilir
    def processed_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield preprocessor.process_frame(frame)

    frame_features = segmentor.segment_video(processed_frames())
    cap.release()
    # Tum mesafeler (temas, savrulma, ivme) bu carpanla "referans piksele" cevrildi:
    # 26 referans piksel = bir bas yaricapi (bkz. segmentor.py, OLCEK NORMALIZASYONU)
    info["scale_factor"] = segmentor.scale_factor
    if verbose:
        print(f"  Olcek faktoru: {segmentor.scale_factor:.2f}")

    # Temas ani icin her karenin bas/ayak konumlari (maskeler olmadan, bellek dostu)
    moment_features = []

    for seg_results in frame_features:
        moment_features.append({
            key: {k: v for k, v in seg_results.get(key, {}).items() if k != "mask" and k != "contours"}
            for key in ("red_helmet", "blue_helmet", "red_foot", "blue_foot")
        })

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
            # Pair anahtarlarini temasa ekle
            contact["helmet_key"] = helmet_key
            contact["foot_key"] = foot_key

            if best_contact is None:
                best_contact = contact
                continue

            # Karşılaştırma mantığı:
            # 1. Eğer birinde çakışma pikseli daha fazlaysa o tercih edilir.
            if contact["overlap_pixels"] > best_contact["overlap_pixels"]:
                best_contact = contact
            elif contact["overlap_pixels"] == best_contact["overlap_pixels"]:
                # 2. Çakışma pikselleri eşitse (örn. ikisi de 0), kontur mesafesi daha yakın olan çift seçilir.
                if contact["min_contour_distance"] < best_contact["min_contour_distance"]:
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
                "helmet_key": None,
                "foot_key": None,
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

    if verbose:
        print(f"  Toplam islenen kare: {frame_idx}")

    # -- Temas Ani --
    contact_moment = find_contact_moment(moment_features, info["fps"], user_frame=contact_frame)
    if verbose:
        if contact_moment["frame"] is None:
            print("  Temas ani: bulunamadi")
        else:
            print(f"  Temas ani: kare {contact_moment['frame']} ({contact_moment['time_sec']:.2f} sn) - "
                  f"{contact_moment['method_tr']}, hedef: {contact_moment['target']}")

    # Olcumler temas anina SADECE an guvenilirse odaklanir: kullanicinin isaretledigi kare ya da
    # ayagin basa gercekten yaklastigi kare. Otomatik "basin en hizli hareket ettigi kare" yontemi
    # kliplerin yaklasik yarisinda yanlis kareye dusuyor; o kareye odaklanmak gercek tepkiyi
    # pencerenin disinda birakip ayirt ediciligi dusuruyordu (bas donusu AUC %69 -> %52).
    reliable_moment = contact_moment["method"] in (METHOD_USER, METHOD_APPROACH)
    analysis_anchor = contact_moment["frame"] if reliable_moment else None
    analysis_target = contact_moment["target"] if reliable_moment else None
    contact_moment["used_for_analysis"] = reliable_moment

    # -- Bas Tepkisi (govdeye gore savrulma, bas donusu) --
    head_response = compute_head_response(moment_features, info["fps"],
                                          anchor_frame=analysis_anchor,
                                          target=analysis_target)
    if verbose:
        hr = head_response["max"]
        print(f"  Bas tepkisi: govdeye gore savrulma {hr['relative_swing']:.1f} px, "
              f"bas donusu {hr['rotation']:.2f} yaricap")

    # -- Temas Ozeti --
    contact_summary = contact_analyzer.analyze_video_contacts(frame_contacts)

    # -- Kinematik Analiz --
    # Hedef kask: temas aninda basi hedef alinan sporcu. Bilinmiyorsa (eski yontem)
    # temasin veya en yakin gecisin gerceklestigi karedeki kask.
    target_helmet_key = f"{contact_moment['target']}_helmet" if contact_moment["target"] else None
    
    # 1. Eger fiziksel temas varsa, en yuksek overlap olan karedeki kaski sec
    peak_overlap = 0
    for r in (frame_contacts if target_helmet_key is None else []):
        if r.get("contact_type") == "overlap" and r.get("overlap_pixels", 0) > peak_overlap:
            peak_overlap = r["overlap_pixels"]
            target_helmet_key = r.get("helmet_key")

    # 2. Eger fiziksel temas yoksa, en yakin gecisin (min contour distance) oldugu karedeki kaski sec
    if target_helmet_key is None:
        min_dist = float("inf")
        for r in frame_contacts:
            dist = r.get("min_contour_distance", float("inf"))
            if dist < min_dist:
                min_dist = dist
                target_helmet_key = r.get("helmet_key")

    # Geriye donuk varsayilan secim
    if target_helmet_key is None:
        target_helmet_key = "red_helmet"

    def helmet_trajectory(helmet_key):
        """Kaskin (burun) gercekten olculdugu karelerden yorunge: [(x, y, kare), ...]"""
        traj = []
        for i, mf in enumerate(moment_features):
            h = mf.get(helmet_key, {})
            if h.get("centroids") and not h.get("held"):
                x, y = h["centroids"][0]
                traj.append((x, y, i))
        return traj

    # Sporcunun kimligi kopsa bile yorunge bolunmez (eskiden en uzun parca aliniyordu)
    best_traj_key = target_helmet_key
    best_trajectory = helmet_trajectory(target_helmet_key)
    if len(best_trajectory) < 3:
        other_key = "blue_helmet" if target_helmet_key == "red_helmet" else "red_helmet"
        other = helmet_trajectory(other_key)
        if len(other) > len(best_trajectory):
            best_trajectory, best_traj_key = other, other_key

    kinematic_result = kinematic_analyzer.classify_motion(best_trajectory,
                                                          anchor_frame=analysis_anchor)

    # Ivme grafigi olustur
    graph_path = ""
    if save_graph and len(best_trajectory) >= 3:
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
        "contact_moment": contact_moment,
        "head_response": head_response,
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