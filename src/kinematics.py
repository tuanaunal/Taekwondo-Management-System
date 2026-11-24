"""
kinematics.py — Kinematik Veri Çıkarımı ve Fiziksel Modelleme
===============================================================
Nesne yörüngelerinden yer değiştirme, hız ve ivme hesaplar.
İvme profilini analiz ederek "Darbe İvmesi" ve "Aktif Kaçış Refleksi"
ayrımını matematiksel olarak yapar.
"""

import numpy as np
from scipy.signal import savgol_filter
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

REFERENCE_FPS = 30.0  # Kare tabanlı eşiklerin ayarlandığı FPS

# Savitzky–Golay filtresi: her noktanın çevresindeki kısa bir pencereye 2. derece polinom
# oturtur. Hız ve ivme bu polinomun türevinden doğrudan hesaplanır; böylece gürültülü
# koordinatın iki kez farkı alınmaz (fark almak gürültüyü katlıyordu).
# Spor biyomekaniğinde yaygın yöntemdir: titremeyi bastırır, ani hareketi (darbe) korur.
SAVGOL_WINDOW_SEC = 0.2   # Pencere süresi (30 FPS'te 7 kare)
SAVGOL_POLYORDER = 2

# Temas anına odaklanma: ivme ve savrulma, temas anının etrafındaki pencereden ölçülür.
# Böylece klipteki başka hareketler (adım atma, başka bir tekme) sonucu bozmaz.
ANCHOR_WINDOW_SEC = 0.5      # Temas anının ± bu kadar saniyesi içindeki ivme incelenir
ANCHOR_TOLERANCE_SEC = 0.17  # Temas anı tahmini ± bu kadar kayabilir (30 FPS'te 5 kare)
SWING_HALF_SEC = 0.33        # Savrulma: ±0.33 sn pencerenin başı ile sonu arasındaki fark


class KinematicAnalyzer:
    """
    Nesne yörüngelerinden kinematik büyüklükleri hesaplar
    ve ivme profilini sınıflandırır.
    """

    def __init__(self, fps: float = TARGET_FPS):
        self.fps = fps
        self.dt = 1.0 / fps  # Kare aralığı (saniye)
        # Eşikler 30 FPS'te ayarlandı. Kare sayısıyla ifade edilen sabitler bu
        # oranla ölçeklenerek farklı FPS'lerde aynı süreye/hıza denk gelir.
        self.frame_scale = fps / REFERENCE_FPS

    def _despike(self, positions: np.ndarray) -> np.ndarray:
        """
        Tek karelik hatalı sıçramaları düzeltir.
        YOLO bazen tek bir karede kafa yerine başka bir noktayı (ör. ayak) verir:
        konum bir karede uzağa sıçrar ve bir sonraki karede geri döner.
        Sadece bu "gidip geri dönen" tekil sıçramalar komşuların ortalamasıyla
        değiştirilir. Kalıcı yer değiştirme (gerçek darbede kafanın savrulması)
        korunur; aksi halde takip donar ve darbe hareketsiz görünür.
        """
        despiked = np.copy(positions)
        # 30 FPS'te kare başına 50 px; FPS arttıkça kare başına izin verilen
        # hareket aynı hıza denk gelecek şekilde azalır.
        MAX_JUMP = 50.0 / self.frame_scale
        for i in range(1, len(positions) - 1):
            jump_out = np.linalg.norm(positions[i] - positions[i - 1])
            jump_back = np.linalg.norm(positions[i + 1] - positions[i - 1])
            if jump_out > MAX_JUMP and jump_back <= MAX_JUMP:
                despiked[i] = (positions[i - 1] + positions[i + 1]) / 2.0
        return despiked

    def _savgol_window(self, n: int) -> int:
        """Pencere boyu: SAVGOL_WINDOW_SEC'e denk gelen tek sayılı kare sayısı (veri boyunu aşmaz)."""
        w = int(round(SAVGOL_WINDOW_SEC * self.fps))
        w = max(w, SAVGOL_POLYORDER + 2)
        if w % 2 == 0:
            w += 1
        if w > n:
            w = n if n % 2 == 1 else n - 1
        return w

    def kinematic_series(self, trajectory: list) -> dict:
        """
        Yörüngeden yumuşatılmış konum, hız ve ivme serilerini hesaplar.

        1. Kareler eşit aralıklı olsun diye eksik kareler doğrusal ara değerle doldurulur.
        2. Tek karelik hatalı sıçramalar düzeltilir (_despike).
        3. Savitzky–Golay filtresiyle konum yumuşatılır; hız ve ivme vektörleri aynı
           polinomun 1. ve 2. türevinden hesaplanır.

        Returns
        -------
        dict
            raw: (n, 2) ara değerli + sıçrama düzeltilmiş konum (yumuşatılmamış)
            position: (n, 2) yumuşatılmış konum
            velocity: (n, 2) hız vektörü (px/sn)
            acceleration: (n, 2) ivme vektörü (px/sn²)
            frames: (n,) kare numaraları
        """
        if len(trajectory) == 0:
            empty = np.zeros((0, 2))
            return {"raw": empty, "position": empty, "velocity": empty,
                    "acceleration": empty, "frames": np.zeros(0)}

        frames = np.array([t[2] for t in trajectory], dtype=float)
        xy = np.array([(t[0], t[1]) for t in trajectory], dtype=float)
        order = np.argsort(frames)
        frames, xy = frames[order], xy[order]
        frames, unique_idx = np.unique(frames, return_index=True)
        xy = xy[unique_idx]

        grid = np.arange(frames[0], frames[-1] + 1)
        raw = np.column_stack([np.interp(grid, frames, xy[:, 0]), np.interp(grid, frames, xy[:, 1])])
        raw = self._despike(raw)
        n = len(raw)

        w = self._savgol_window(n)
        if n < SAVGOL_POLYORDER + 2 or w <= SAVGOL_POLYORDER:
            # Çok kısa yörünge: filtre uygulanamaz, basit farkla devam edilir
            vel = np.gradient(raw, self.dt, axis=0) if n > 1 else np.zeros_like(raw)
            acc = np.gradient(vel, self.dt, axis=0) if n > 2 else np.zeros_like(raw)
            return {"raw": raw, "position": raw, "velocity": vel, "acceleration": acc, "frames": grid}

        pos = savgol_filter(raw, w, SAVGOL_POLYORDER, axis=0, mode="interp")
        vel = savgol_filter(raw, w, SAVGOL_POLYORDER, deriv=1, delta=self.dt, axis=0, mode="interp")
        acc = savgol_filter(raw, w, SAVGOL_POLYORDER, deriv=2, delta=self.dt, axis=0, mode="interp")
        return {"raw": raw, "position": pos, "velocity": vel, "acceleration": acc, "frames": grid}

    def smooth_positions(self, positions: np.ndarray, window_size: int = None) -> np.ndarray:
        """Eşit aralıklı konum dizisini (n, 2) sıçrama düzeltme + Savitzky–Golay ile yumuşatır."""
        positions = np.asarray(positions, dtype=float)
        traj = [(x, y, i) for i, (x, y) in enumerate(positions)]
        return self.kinematic_series(traj)["position"]

    # ────────────────────────────────────
    # YER DEĞİŞTİRME / HIZ / İVME
    # ────────────────────────────────────
    def compute_displacement(self, trajectory: list) -> np.ndarray:
        """Ardışık kareler arasındaki (yumuşatılmış) yer değiştirme, piksel."""
        if len(trajectory) < 2:
            return np.array([0.0])
        pos = self.kinematic_series(trajectory)["position"]
        return np.linalg.norm(np.diff(pos, axis=0), axis=1)

    def compute_velocity(self, trajectory: list) -> np.ndarray:
        """Hız büyüklüğü (px/sn), Savitzky–Golay 1. türevinden."""
        if len(trajectory) < 2:
            return np.array([0.0])
        return np.linalg.norm(self.kinematic_series(trajectory)["velocity"], axis=1)

    def compute_acceleration(self, trajectory: list) -> np.ndarray:
        """İvme büyüklüğü (px/sn²), Savitzky–Golay 2. türevinden."""
        if len(trajectory) < 3:
            return np.array([0.0])
        return np.linalg.norm(self.kinematic_series(trajectory)["acceleration"], axis=1)

    # ────────────────────────────────────
    # İVME PROFİLİ SINIFLANDIRMA
    # ────────────────────────────────────
    def classify_motion(self, trajectory: list, anchor_frame: int = None) -> dict:
        """
        İvme profilini analiz ederek hareket tipini sınıflandırır.

        anchor_frame verilirse (temas anı), ivme ve savrulma bu anın etrafındaki
        pencereden ölçülür; verilmezse klibin tamamı kullanılır.

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
                'peak_frame': int,           # ivme tepesinin KARE NUMARASI (videodaki)
                'anchor_frame': int | None,  # kullanılan temas anı
                'confidence': float,
                'details': str,
            }
        """
        displacement = self.compute_displacement(trajectory)
        velocity = self.compute_velocity(trajectory)
        acceleration = self.compute_acceleration(trajectory)
        series = self.kinematic_series(trajectory)
        grid = series["frames"]

        result = {
            "displacement_profile": displacement,
            "velocity_profile": velocity,
            "acceleration_profile": acceleration,
            "max_acceleration": 0.0,
            "net_displacement": 0.0,
            "peak_frame": 0,
            "anchor_frame": None,
            "type": "stationary",
            "confidence": 0.0,
            "details": "",
        }

        if len(acceleration) == 0 or np.max(np.abs(acceleration)) < 1.0:
            result["type"] = "stationary"
            result["confidence"] = 0.9
            result["details"] = "Nesne neredeyse hareketsiz."
            return result

        # ── İnceleme penceresi: temas anının etrafı (yoksa klibin tamamı) ──
        abs_acc_all = np.abs(acceleration)
        if (anchor_frame is not None and len(grid) == len(abs_acc_all)
                and grid[0] <= anchor_frame <= grid[-1]):
            w = max(1, int(round(ANCHOR_WINDOW_SEC * self.fps)))
            in_win = (grid >= anchor_frame - w) & (grid <= anchor_frame + w)
            result["anchor_frame"] = int(anchor_frame)
        else:
            in_win = np.ones(len(abs_acc_all), dtype=bool)
            anchor_frame = None
        win_idx = np.where(in_win)[0]
        abs_acc = abs_acc_all[win_idx]

        max_acc = float(np.max(abs_acc))
        peak_idx = int(win_idx[np.argmax(abs_acc)])  # tüm seri içindeki indeks
        result["max_acceleration"] = max_acc
        # Kare numarası olarak (eskiden dizi indeksiydi ve yörünge dizisinde kaymaya yol açıyordu)
        result["peak_frame"] = int(grid[peak_idx]) if len(grid) == len(abs_acc_all) else peak_idx
        
        # ── Net savrulma: ±0.33 sn pencerenin başı ile sonu arasındaki yer değiştirme ──
        # Temas anı varsa pencere merkezi temas anının ±0.17 sn çevresinde aranır (tahmin payı);
        # yoksa ivme tepesi kullanılır. Konum: eşit aralıklı ızgaradaki yumuşatılmamış konum.
        raw_pos = series["raw"]
        half = max(1, int(round(SWING_HALF_SEC * self.fps)))
        if anchor_frame is not None:
            tol = max(0, int(round(ANCHOR_TOLERANCE_SEC * self.fps)))
            a = int(anchor_frame - grid[0])
            centers = range(a - tol, a + tol + 1)
        else:
            centers = [peak_idx]
        net_disp = 0.0
        for c in centers:
            if c - half >= 0 and c + half < len(raw_pos):
                net_disp = max(net_disp, float(np.linalg.norm(raw_pos[c + half] - raw_pos[c - half])))
            elif len(raw_pos) > 1 and anchor_frame is None:
                s0, s1 = max(0, c - half), min(len(raw_pos) - 1, c + half)
                net_disp = max(net_disp, float(np.linalg.norm(raw_pos[s1] - raw_pos[s0])))
        
        result["net_displacement"] = net_disp

        # ── Kriter 1: Tepe ivme eşiği ──
        exceeds_threshold = max_acc > ACCELERATION_IMPACT_THRESHOLD

        # ── Kriter 2: İvme süresi (kaç ARDIŞIK kare boyunca yüksek?) ──
        # Eskiden eşiği geçen tüm kareler (ardışık olmasa da) sayılıyordu; artık en uzun kesintisiz süre.
        high_acc_mask = abs_acc > (ACCELERATION_IMPACT_THRESHOLD * 0.5)
        high_acc_duration, run = 0, 0
        for flag in high_acc_mask:
            run = run + 1 if flag else 0
            high_acc_duration = max(high_acc_duration, run)
        max_impact_frames = IMPACT_DURATION_MAX_FRAMES * self.frame_scale
        is_short_duration = 0 < high_acc_duration <= max_impact_frames

        # ── Kriter 3: İvme varyasyon katsayısı ──
        mean_acc = np.mean(abs_acc) if np.mean(abs_acc) > 0 else 1e-6
        cv = np.std(abs_acc) / mean_acc  # Coefficient of Variation
        is_sharp = cv > EVASION_SMOOTHNESS_THRESHOLD
        result["acceleration_cv"] = float(cv)

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
