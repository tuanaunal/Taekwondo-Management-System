"""
kinematics.py — Kinematik Veri Çıkarımı ve Fiziksel Modelleme
===============================================================
Nesne yörüngelerinden yer değiştirme, hız ve ivme hesaplar.
İvme profilini analiz ederek "Darbe İvmesi" ve "Aktif Kaçış Refleksi"
ayrımını matematiksel olarak yapar.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan grafik oluşturma
import matplotlib.pyplot as plt
from src.config import (
    TARGET_FPS,
    ACCELERATION_IMPACT_THRESHOLD,
    IMPACT_DURATION_MAX_FRAMES,
    EVASION_SMOOTHNESS_THRESHOLD,
    GRAPHS_DIR,
)
import os


class KinematicAnalyzer:
    """
    Nesne yörüngelerinden kinematik büyüklükleri hesaplar
    ve ivme profilini sınıflandırır.
    """

    def __init__(self, fps: float = TARGET_FPS):
        self.fps = fps
        self.dt = 1.0 / fps  # Kare aralığı (saniye)

    def smooth_positions(self, positions: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Yörünge koordinatlarındaki yüksek frekanslı titremeleri (jitter)
        azaltmak için hareketli ortalama (moving average) filtresi uygular.
        Ayrıca YOLO'nun imkansız sıçramalarını (1 karede >50 piksel) engeller.
        """
        if len(positions) < window_size:
            return positions
            
        # 1. Despike (Fiziksel olarak imkansız sıçramaları filtrele)
        # Bir insan kafası 1 karede (33ms) 50 pikselden fazla hareket edemez.
        # Eğer ederse, bu YOLO'nun kafa yerine ayağı takip ettiğinin kanıtıdır.
        despiked = np.copy(positions)
        MAX_JUMP = 50.0
        for i in range(1, len(despiked)):
            dist = np.linalg.norm(despiked[i] - despiked[i-1])
            if dist > MAX_JUMP:
                # İmkansız sıçrama: Bir önceki geçerli konumu koru
                despiked[i] = despiked[i-1]
                
        # 2. Hareketli Ortalama (Moving Average)
        smoothed = np.copy(despiked)
        half_w = window_size // 2
        for i in range(len(despiked)):
            start = max(0, i - half_w)
            end = min(len(despiked), i + half_w + 1)
            smoothed[i] = np.mean(despiked[start:end], axis=0)
            
        return smoothed

    # ────────────────────────────────────
    # YER DEĞİŞTİRME HESABI
    # ────────────────────────────────────
    def compute_displacement(self, trajectory: list) -> np.ndarray:
        """
        Ardışık noktalar arasındaki piksel yer değiştirmesini hesaplar.
        """
        if len(trajectory) < 2:
            return np.array([0.0])

        positions = np.array([(t[0], t[1]) for t in trajectory], dtype=float)
        positions = self.smooth_positions(positions, window_size=3)
        diff = np.diff(positions, axis=0)
        displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        return displacement

    def compute_velocity(self, trajectory: list) -> np.ndarray:
        """
        Hız = ΔDisplacement / Δt (frame_idx farkı kullanılarak)
        """
        if len(trajectory) < 2:
            return np.array([0.0])
            
        displacement = self.compute_displacement(trajectory)
        
        # Frame farklarını hesapla
        frames = np.array([t[2] for t in trajectory], dtype=float)
        frame_diffs = np.diff(frames)
        
        # 0'a bölme hatasını önle
        frame_diffs[frame_diffs == 0] = 1.0
        
        # Her bir aralık için dt = frame_diff * self.dt
        actual_dts = frame_diffs * self.dt
        
        velocity = displacement / actual_dts
        return velocity

    def compute_acceleration(self, trajectory: list) -> np.ndarray:
        """
        İvme = ΔVelocity / Δt
        """
        velocity = self.compute_velocity(trajectory)
        if len(velocity) < 2:
            return np.array([0.0])

        frames = np.array([t[2] for t in trajectory], dtype=float)
        # İvme için ardışık hızların zaman farkı, aslında noktalar arası orta noktaların zaman farkıdır
        # Basitlik için hızlar arasındaki frame farkını, n. ve n+1. noktalar arasındaki fark olarak alıyoruz
        frame_diffs = np.diff(frames[1:]) # length is len(velocity) - 1
        frame_diffs[frame_diffs == 0] = 1.0
        actual_dts = frame_diffs * self.dt

        acceleration = np.diff(velocity) / actual_dts
        return acceleration

    # ────────────────────────────────────
    # İVME PROFİLİ SINIFLANDIRMA
    # ────────────────────────────────────
    def classify_motion(self, trajectory: list) -> dict:
        """
        İvme profilini analiz ederek hareket tipini sınıflandırır.

        Darbe İvmesi Özellikleri:
          - Ani, keskin ivme artışı (yüksek tepe değeri)
          - Kısa süre (birkaç frame içinde tamamlanır)
          - İvme varyasyon katsayısı yüksek

        Aktif Kaçış Özellikleri:
          - Kademeli ivme artışı
          - Daha uzun süre
          - İvme varyasyon katsayısı düşük

        Returns
        -------
        dict
            {
                'type': 'impact' | 'evasion' | 'stationary',
                'max_acceleration': float,
                'acceleration_profile': np.ndarray,
                'velocity_profile': np.ndarray,
                'displacement_profile': np.ndarray,
                'peak_frame': int,
                'confidence': float,
                'details': str,
            }
        """
        displacement = self.compute_displacement(trajectory)
        velocity = self.compute_velocity(trajectory)
        acceleration = self.compute_acceleration(trajectory)

        result = {
            "displacement_profile": displacement,
            "velocity_profile": velocity,
            "acceleration_profile": acceleration,
            "max_acceleration": 0.0,
            "peak_frame": 0,
            "type": "stationary",
            "confidence": 0.0,
            "details": "",
        }

        if len(acceleration) == 0 or np.max(np.abs(acceleration)) < 1.0:
            result["type"] = "stationary"
            result["confidence"] = 0.9
            result["details"] = "Nesne neredeyse hareketsiz."
            return result

        abs_acc = np.abs(acceleration)
        max_acc = np.max(abs_acc)
        peak_idx = np.argmax(abs_acc)
        result["max_acceleration"] = max_acc
        result["peak_frame"] = peak_idx
        
        # ── Net Displacement (Ghost Hit vs Real Hit ayrımı için) ──
        # İvme zirvesinden 10 kare öncesi ve 10 kare sonrası arasındaki net yer değiştirme.
        # Gerçek darbede kafa geriye savrulur (net_disp > 165px). 
        # Ghost hitte kafa yerine döner veya az hareket eder (net_disp < 165px).
        net_disp = 0.0
        if len(trajectory) > 20:
            start_idx = max(0, peak_idx - 10)
            end_idx = min(len(trajectory) - 1, peak_idx + 10)
            p_start = np.array([trajectory[start_idx][0], trajectory[start_idx][1]])
            p_end = np.array([trajectory[end_idx][0], trajectory[end_idx][1]])
            net_disp = np.linalg.norm(p_end - p_start)
        
        result["net_displacement"] = net_disp

        # ── Kriter 1: Tepe ivme eşiği ──
        exceeds_threshold = max_acc > ACCELERATION_IMPACT_THRESHOLD

        # ── Kriter 2: İvme süresi (kaç frame boyunca yüksek?) ──
        high_acc_mask = abs_acc > (ACCELERATION_IMPACT_THRESHOLD * 0.5)
        high_acc_duration = np.sum(high_acc_mask)
        is_short_duration = 0 < high_acc_duration <= IMPACT_DURATION_MAX_FRAMES

        # ── Kriter 3: İvme varyasyon katsayısı ──
        mean_acc = np.mean(abs_acc) if np.mean(abs_acc) > 0 else 1e-6
        cv = np.std(abs_acc) / mean_acc  # Coefficient of Variation
        is_sharp = cv > EVASION_SMOOTHNESS_THRESHOLD

        # ── Karar ──
        impact_score = 0.0
        if exceeds_threshold:
            impact_score += 0.4
        if is_short_duration:
            impact_score += 0.3
        if is_sharp:
            impact_score += 0.3

        if impact_score >= 0.6:
            result["type"] = "impact"
            result["confidence"] = impact_score
            result["details"] = (
                f"Darbe ivmesi tespit edildi. "
                f"Tepe: {max_acc:.1f} px/s², "
                f"Süre: {high_acc_duration} frame, "
                f"CV: {cv:.2f}"
            )
        else:
            result["type"] = "evasion"
            result["confidence"] = 1.0 - impact_score
            result["details"] = (
                f"Aktif kaçış hareketi tespit edildi. "
                f"Tepe: {max_acc:.1f} px/s², "
                f"Süre: {high_acc_duration} frame, "
                f"CV: {cv:.2f}"
            )

        return result

    # ────────────────────────────────────
    # İVME-ZAMAN GRAFİĞİ
    # ────────────────────────────────────
    def plot_kinematics(
        self, trajectory: list, title: str = "Kinematik Analiz",
        save_path: str = None
    ) -> str:
        """
        Yer değiştirme, hız ve ivme grafiklerini çizer.

        Parameters
        ----------
        trajectory : list of tuple
            [(cx, cy, frame_idx), ...]
        title : str
            Grafik başlığı.
        save_path : str, optional
            Kaydedilecek dosya yolu. None ise otomatik oluşturulur.

        Returns
        -------
        str
            Kaydedilen grafik dosya yolu.
        """
        displacement = self.compute_displacement(trajectory)
        velocity = self.compute_velocity(trajectory)
        acceleration = self.compute_acceleration(trajectory)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=100)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Zaman eksenleri
        t_disp = np.arange(len(displacement)) / self.fps
        t_vel = np.arange(len(velocity)) / self.fps
        t_acc = np.arange(len(acceleration)) / self.fps

        # ── Yer Değiştirme ──
        axes[0].plot(t_disp, displacement, "b-o", markersize=4, linewidth=1.5)
        axes[0].set_ylabel("Yer Değiştirme\n(piksel)", fontsize=10)
        axes[0].set_title("Kare Bazlı Yer Değiştirme", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(t_disp, displacement, alpha=0.2, color="blue")

        # ── Hız ──
        axes[1].plot(t_vel, velocity, "g-o", markersize=4, linewidth=1.5)
        axes[1].set_ylabel("Hız\n(px/s)", fontsize=10)
        axes[1].set_title("Hız Profili", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(t_vel, velocity, alpha=0.2, color="green")

        # ── İvme ──
        axes[2].plot(t_acc, acceleration, "r-o", markersize=4, linewidth=1.5)
        axes[2].axhline(
            y=ACCELERATION_IMPACT_THRESHOLD,
            color="orange", linestyle="--", linewidth=1,
            label=f"Darbe Eşiği ({ACCELERATION_IMPACT_THRESHOLD} px/s²)",
        )
        axes[2].axhline(
            y=-ACCELERATION_IMPACT_THRESHOLD,
            color="orange", linestyle="--", linewidth=1,
        )
        axes[2].set_ylabel("İvme\n(px/s²)", fontsize=10)
        axes[2].set_xlabel("Zaman (saniye)", fontsize=10)
        axes[2].set_title("İvme-Zaman Grafiği", fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(t_acc, acceleration, alpha=0.2, color="red")
        axes[2].legend(fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                GRAPHS_DIR,
                title.replace(" ", "_").replace("/", "_") + ".png",
            )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        return save_path
