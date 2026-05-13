"""
preprocessor.py — Video Ön-İşleme Pipeline
============================================
Her video karesine sırasıyla uygulanan ön-işleme adımları:
  1. CLAHE ile ışık dengelemesi
  2. Gaussian / Median filtreleme ile gürültü temizliği
  3. Dinamik ROI ile aktif bölge maskelemesi
"""

import cv2
import numpy as np
from src.config import (
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE,
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
    ROI_MARGIN, TARGET_FPS, TARGET_WIDTH, TARGET_HEIGHT,
)


class VideoPreprocessor:
    """
    Video karelerini analiz için hazırlayan ön-işleme sınıfı.
    """

    def __init__(self):
        # CLAHE nesnesi (LAB-L kanalı için)
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE,
        )
        # Arka plan çıkarıcı (Dinamik ROI için)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self.roi_bbox = None  # Mevcut ROI sınır kutusu

    # ────────────────────────────────────
    # ANA İŞLEM
    # ────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Tek bir kareye tüm ön-işleme adımlarını uygular.

        Parameters
        ----------
        frame : np.ndarray
            BGR formatında ham video karesi.

        Returns
        -------
        np.ndarray
            İşlenmiş (ışık dengeli, filtrelenmiş) BGR kare.
        """
        # 1. Çözünürlük standardizasyonu
        frame = self._standardize_resolution(frame)

        # 2. CLAHE ile ışık dengeleme
        frame = self._apply_clahe(frame)

        # 3. Gaussian Blur
        frame = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, 0)

        # 4. Median Blur
        frame = cv2.medianBlur(frame, MEDIAN_KERNEL_SIZE)

        return frame

    # ────────────────────────────────────
    # ÇÖZÜNÜRLÜK STANDARDİZASYONU
    # ────────────────────────────────────
    def _standardize_resolution(self, frame: np.ndarray) -> np.ndarray:
        """Kareyi hedef çözünürlüğe ölçekler."""
        h, w = frame.shape[:2]
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            frame = cv2.resize(
                frame,
                (TARGET_WIDTH, TARGET_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    # ────────────────────────────────────
    # CLAHE — Adaptif Histogram Eşitleme
    # ────────────────────────────────────
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR → LAB dönüşümü yaparak L kanalına CLAHE uygular.
        Spot ışıkları kaynaklı parlamaları dengeler.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # L kanalına CLAHE uygula
        l_enhanced = self.clahe.apply(l_channel)

        # Kanalları birleştir ve BGR'ye dön
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return result

    # ────────────────────────────────────
    # DİNAMİK ROI — Aktif Bölge Tespiti
    # ────────────────────────────────────
    def compute_roi(self, frame: np.ndarray) -> tuple:
        """
        Arka plan çıkarma (MOG2) ile aktif mücadele bölgesini tespit eder.

        Returns
        -------
        tuple
            (x, y, w, h) — ROI sınır kutusu. Tespit edilemezse tam kare döner.
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Gürültü temizliği
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Ön plan piksellerinin sınır kutusunu bul
        coords = cv2.findNonZero(fg_mask)
        if coords is not None and len(coords) > 100:
            x, y, w, h = cv2.boundingRect(coords)

            # Marjin ekle
            img_h, img_w = frame.shape[:2]
            x = max(0, x - ROI_MARGIN)
            y = max(0, y - ROI_MARGIN)
            w = min(img_w - x, w + 2 * ROI_MARGIN)
            h = min(img_h - y, h + 2 * ROI_MARGIN)

            self.roi_bbox = (x, y, w, h)
        else:
            # Tüm kareyi kullan
            self.roi_bbox = (0, 0, frame.shape[1], frame.shape[0])

        return self.roi_bbox

    def apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI sınır kutusu içindeki bölgeyi kırpar."""
        if self.roi_bbox is None:
            return frame
        x, y, w, h = self.roi_bbox
        return frame[y:y + h, x:x + w]

    def reset(self):
        """Arka plan modelini ve ROI'yi sıfırlar (yeni video için)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self.roi_bbox = None


def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Video dosyasını açar ve VideoCapture nesnesini döner.

    Raises
    ------
    FileNotFoundError
        Video dosyası bulunamazsa.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """VideoCapture nesnesinden video meta bilgilerini döner."""
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
