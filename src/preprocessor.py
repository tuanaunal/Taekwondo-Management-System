"""
preprocessor.py — Video Ön-İşleme Pipeline
============================================
Her video karesine sırasıyla uygulanan ön-işleme adımları:
  1. Çözünürlük standardizasyonu (1920x1080)
  2. CLAHE ile ışık dengelemesi

Not: Eskiden YOLO'dan önce Gaussian + Median bulanıklaştırma da yapılıyordu.
Bulanıklaştırma hızlı tekmelerde ayak bileğinin kaybolmasına yol açtığı ve
YOLO gürültüye zaten dayanıklı olduğu için kaldırıldı. Kullanılmayan MOG2
arka plan çıkarma / ROI kodu da kaldırıldı.
"""

import cv2
import numpy as np
from src.config import (
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    TARGET_FPS, TARGET_WIDTH, TARGET_HEIGHT,
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
            İşlenmiş (ışık dengeli) BGR kare.
        """
        # 1. Çözünürlük standardizasyonu
        # Not: Temas ve savrulma eşikleri şimdilik piksel cinsinden ve 1080p'ye
        # göre ayarlı olduğu için bu adım korunuyor. Mesafeler sporcu boyuna
        # göre ölçeklendiğinde (yol haritası 10. adım) kaldırılacak.
        frame = self._standardize_resolution(frame)

        # 2. CLAHE ile ışık dengeleme
        frame = self._apply_clahe(frame)

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
    """
    VideoCapture nesnesinden video meta bilgilerini döner.

    Bazı video dosyaları FPS bilgisini 0 veya geçersiz döndürür; bu durumda
    varsayılan TARGET_FPS kullanılır ve "fps_estimated" True olur.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_estimated = not (fps and fps == fps and 1.0 <= fps <= 1000.0)  # fps == fps: NaN kontrolü
    if fps_estimated:
        fps = float(TARGET_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "fps": fps,
        "fps_estimated": fps_estimated,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
    }
