"""
video_player.py — OpenCV Tabanlı Video Oynatıcı Widget
========================================================
PyQt5 QLabel üzerinde video karelerini gösterir.
Frame-by-frame ileri/geri, yavaş çekim ve
segmentasyon overlay desteği sunar.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class VideoPlayerWidget(QWidget):
    """
    OpenCV + QLabel tabanlı video oynatıcı.

    Signals
    -------
    frame_changed(int)
        Mevcut kare değiştiğinde frame indeksini yayar.
    playback_finished()
        Video sonuna ulaşıldığında yayar.
    """

    frame_changed = pyqtSignal(int)
    playback_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.frames = []          # Tüm kareleri bellekte tut (kısa videolar)
        self.overlay_frames = []  # Segmentasyon overlay kareleri
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playing = False
        self.show_overlay = False

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video gösterim alanı
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
                min-height: 400px;
            }
        """)
        self.video_label.setText("Video yüklemek için 'Dosya Aç' butonunu kullanın")
        self.video_label.setStyleSheet(self.video_label.styleSheet() + """
            QLabel { color: #8899aa; font-size: 14px; }
        """)
        layout.addWidget(self.video_label, 1)

        # Frame slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 6px;
                background: #16213e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e94560;
                border: 2px solid #e94560;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #e94560;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.slider)

        # Kontrol butonları
        controls = QHBoxLayout()

        self.btn_prev = QPushButton("⏮ Önceki")
        self.btn_play = QPushButton("▶ Oynat")
        self.btn_next = QPushButton("Sonraki ⏭")
        self.btn_overlay = QPushButton("🎨 Overlay")
        self.frame_info = QLabel("Kare: 0/0")

        btn_style = """
            QPushButton {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #0f3460;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0f3460;
                border-color: #e94560;
            }
            QPushButton:pressed {
                background-color: #e94560;
            }
        """

        for btn in [self.btn_prev, self.btn_play, self.btn_next, self.btn_overlay]:
            btn.setStyleSheet(btn_style)

        self.frame_info.setStyleSheet("""
            QLabel {
                color: #8899aa;
                font-size: 12px;
                padding: 0 10px;
            }
        """)

        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_overlay.clicked.connect(self.toggle_overlay)

        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_next)
        controls.addStretch()
        controls.addWidget(self.btn_overlay)
        controls.addWidget(self.frame_info)

        layout.addLayout(controls)

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._play_next_frame)

    # ────────────────────────────────────
    # VIDEO YÜKLEME
    # ────────────────────────────────────
    def load_video(self, video_path: str):
        """Video dosyasını yükler ve tüm kareleri belleğe alır."""
        self.stop()
        self.frames = []
        self.overlay_frames = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)

        cap.release()
        self.total_frames = len(self.frames)

        if self.total_frames > 0:
            self.slider.setMaximum(self.total_frames - 1)
            self.current_frame = 0
            self._display_frame(0)

        return True

    def set_overlay_frames(self, overlay_frames: list):
        """Segmentasyon overlay karelerini ayarlar."""
        self.overlay_frames = overlay_frames

    # ────────────────────────────────────
    # OYNATMA KONTROLLERİ
    # ────────────────────────────────────
    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if self.total_frames == 0:
            return
        self.playing = True
        self.btn_play.setText("⏸ Durdur")
        self.timer.start(int(1000 / self.fps))

    def pause(self):
        self.playing = False
        self.btn_play.setText("▶ Oynat")
        self.timer.stop()

    def stop(self):
        self.pause()
        self.current_frame = 0
        if self.total_frames > 0:
            self._display_frame(0)
            self.slider.setValue(0)

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._display_frame(self.current_frame)
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._display_frame(self.current_frame)
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)

    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        if self.show_overlay:
            self.btn_overlay.setText("🎨 Overlay: AÇIK")
        else:
            self.btn_overlay.setText("🎨 Overlay: KAPALI")
        self._display_frame(self.current_frame)

    # ────────────────────────────────────
    # KARE GÖSTERİMİ
    # ────────────────────────────────────
    def _display_frame(self, idx: int):
        if idx < 0 or idx >= len(self.frames):
            return

        frame = self.frames[idx].copy()

        # Overlay göster
        if self.show_overlay and idx < len(self.overlay_frames):
            overlay = self.overlay_frames[idx]
            if overlay is not None:
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # BGR → RGB dönüşümü
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # QLabel boyutuna ölçekle
        label_w = self.video_label.width() - 4
        label_h = self.video_label.height() - 4
        if label_w > 0 and label_h > 0:
            scale = min(label_w / w, label_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w > 0 and new_h > 0:
                rgb = cv2.resize(rgb, (new_w, new_h))
                h, w, ch = rgb.shape

        # QImage → QPixmap
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

        # Bilgi güncelle
        time_sec = idx / self.fps if self.fps > 0 else 0
        self.frame_info.setText(
            f"Kare: {idx + 1}/{self.total_frames} | "
            f"Zaman: {time_sec:.2f}s"
        )
        self.frame_changed.emit(idx)

    def _play_next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._display_frame(self.current_frame)
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
        else:
            self.pause()
            self.playback_finished.emit()

    def _on_slider_changed(self, value):
        self.current_frame = value
        self._display_frame(value)
