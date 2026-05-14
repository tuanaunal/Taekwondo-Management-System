"""
video_player.py — OpenCV Tabanlı Video Oynatıcı Widget
========================================================
PyQt5 QLabel üzerinde video karelerini gösterir.
Frame-by-frame ileri/geri, yavaş çekim/hızlandırma,
zoom/pan ve segmentasyon overlay desteği sunar.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QButtonGroup, QFrame,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QCursor


class ZoomableVideoLabel(QLabel):
    """
    Zoom ve Pan destekli video gösterim alanı.
    Mouse wheel ile zoom, mouse drag ile pan yapar.
    """

    zoom_changed = pyqtSignal(float)  # zoom seviyesi değiştiğinde

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_level = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 5.0
        self.zoom_step = 0.25

        # Pan (kaydırma) durumu
        self.pan_offset_x = 0.0  # 0..1 arası normalize değer (merkez)
        self.pan_offset_y = 0.0
        self._panning = False
        self._pan_start = QPoint()
        self._pan_start_offset_x = 0.0
        self._pan_start_offset_y = 0.0

        # Mevcut orijinal kare boyutları
        self._frame_w = 0
        self._frame_h = 0

        self.setMouseTracking(True)

    def wheelEvent(self, event):
        """Mouse scroll ile zoom in/out."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(self.zoom_level + self.zoom_step, self.max_zoom)
        else:
            self.zoom_level = max(self.zoom_level - self.zoom_step, self.min_zoom)

        # Zoom 1x'e düşerse pan'i sıfırla
        if self.zoom_level <= 1.0:
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0

        self.zoom_changed.emit(self.zoom_level)

    def mousePressEvent(self, event):
        """Pan başlat."""
        if event.button() == Qt.LeftButton and self.zoom_level > 1.0:
            self._panning = True
            self._pan_start = event.pos()
            self._pan_start_offset_x = self.pan_offset_x
            self._pan_start_offset_y = self.pan_offset_y
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        """Pan güncelle."""
        if self._panning and self.zoom_level > 1.0:
            delta = event.pos() - self._pan_start
            # Piksel delta'sını normalize et
            if self._frame_w > 0 and self._frame_h > 0:
                dx = -delta.x() / (self._frame_w * self.zoom_level) * 2
                dy = -delta.y() / (self._frame_h * self.zoom_level) * 2
                self.pan_offset_x = max(-1.0, min(1.0, self._pan_start_offset_x + dx))
                self.pan_offset_y = max(-1.0, min(1.0, self._pan_start_offset_y + dy))
                self.zoom_changed.emit(self.zoom_level)
        elif self.zoom_level > 1.0:
            self.setCursor(QCursor(Qt.OpenHandCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        """Pan bitir."""
        if event.button() == Qt.LeftButton:
            self._panning = False
            if self.zoom_level > 1.0:
                self.setCursor(QCursor(Qt.OpenHandCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))

    def reset_zoom(self):
        """Zoom ve pan'i sıfırla."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.zoom_changed.emit(self.zoom_level)

    def zoom_in(self):
        self.zoom_level = min(self.zoom_level + self.zoom_step, self.max_zoom)
        self.zoom_changed.emit(self.zoom_level)

    def zoom_out(self):
        self.zoom_level = max(self.zoom_level - self.zoom_step, self.min_zoom)
        if self.zoom_level <= 1.0:
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0
        self.zoom_changed.emit(self.zoom_level)


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
        self.speed = 1.0          # Oynatma hızı çarpanı

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Video gösterim alanı (Zoomable) ──
        self.video_label = ZoomableVideoLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
                min-height: 450px;
                color: #8899aa;
                font-size: 14px;
            }
        """)
        self.video_label.setText("Video yüklemek için 'Dosya Aç' butonunu kullanın")
        self.video_label.zoom_changed.connect(self._on_zoom_changed)
        layout.addWidget(self.video_label, 1)

        # ── Frame slider ──
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

        # ── Oynatma Kontrolleri ──
        controls = QHBoxLayout()
        controls.setSpacing(6)

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

        # ── Hız ve Zoom Kontrolleri ──
        extra_controls = QHBoxLayout()
        extra_controls.setSpacing(4)

        # Hız kontrolleri
        speed_label = QLabel("⏩ Hız:")
        speed_label.setStyleSheet("color: #8899aa; font-size: 11px; font-weight: bold;")
        extra_controls.addWidget(speed_label)

        self.speed_buttons = {}
        speed_btn_style_normal = """
            QPushButton {
                background-color: #161b22;
                color: #8b949e;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: bold;
                min-width: 42px;
            }
            QPushButton:hover {
                background-color: #21262d;
                border-color: #58a6ff;
                color: #c9d1d9;
            }
        """
        speed_btn_style_active = """
            QPushButton {
                background-color: #1f6feb;
                color: white;
                border: 1px solid #58a6ff;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: bold;
                min-width: 42px;
            }
        """

        for spd in [0.25, 0.5, 1.0, 2.0, 4.0]:
            label = f"{spd}x"
            btn = QPushButton(label)
            btn.setStyleSheet(speed_btn_style_active if spd == 1.0 else speed_btn_style_normal)
            btn.clicked.connect(lambda checked, s=spd: self._set_speed(s))
            btn.setProperty("speed_value", spd)
            self.speed_buttons[spd] = btn
            extra_controls.addWidget(btn)

        # Ayırıcı
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #30363d;")
        extra_controls.addWidget(separator)

        # Zoom kontrolleri
        zoom_label = QLabel("🔍 Zoom:")
        zoom_label.setStyleSheet("color: #8899aa; font-size: 11px; font-weight: bold;")
        extra_controls.addWidget(zoom_label)

        zoom_btn_style = """
            QPushButton {
                background-color: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: bold;
                min-width: 32px;
            }
            QPushButton:hover {
                background-color: #21262d;
                border-color: #e94560;
                color: #e94560;
            }
            QPushButton:pressed {
                background-color: #e94560;
                color: white;
            }
        """

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_reset = QPushButton("1.0x")
        self.btn_zoom_in = QPushButton("+")

        for btn in [self.btn_zoom_out, self.btn_zoom_reset, self.btn_zoom_in]:
            btn.setStyleSheet(zoom_btn_style)

        self.btn_zoom_reset.setStyleSheet(zoom_btn_style + """
            QPushButton {
                min-width: 48px;
                color: #e94560;
            }
        """)

        self.btn_zoom_out.clicked.connect(self.video_label.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.video_label.reset_zoom)
        self.btn_zoom_in.clicked.connect(self.video_label.zoom_in)

        extra_controls.addWidget(self.btn_zoom_out)
        extra_controls.addWidget(self.btn_zoom_reset)
        extra_controls.addWidget(self.btn_zoom_in)

        extra_controls.addStretch()

        layout.addLayout(extra_controls)

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._play_next_frame)

    # ────────────────────────────────────
    # HIZ KONTROLLERİ
    # ────────────────────────────────────
    def _set_speed(self, speed: float):
        """Oynatma hızını ayarlar."""
        self.speed = speed

        # Buton stillerini güncelle
        speed_btn_style_normal = """
            QPushButton {
                background-color: #161b22;
                color: #8b949e;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: bold;
                min-width: 42px;
            }
            QPushButton:hover {
                background-color: #21262d;
                border-color: #58a6ff;
                color: #c9d1d9;
            }
        """
        speed_btn_style_active = """
            QPushButton {
                background-color: #1f6feb;
                color: white;
                border: 1px solid #58a6ff;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: bold;
                min-width: 42px;
            }
        """

        for spd, btn in self.speed_buttons.items():
            btn.setStyleSheet(speed_btn_style_active if spd == speed else speed_btn_style_normal)

        # Eğer oynatılıyorsa timer'ı güncelle
        if self.playing:
            self.timer.stop()
            interval = max(1, int(1000 / self.fps / self.speed))
            self.timer.start(interval)

    # ────────────────────────────────────
    # ZOOM KONTROLLERİ
    # ────────────────────────────────────
    def _on_zoom_changed(self, zoom_level: float):
        """Zoom seviyesi değiştiğinde UI günceller ve kareyi yeniden çizer."""
        self.btn_zoom_reset.setText(f"{zoom_level:.1f}x")
        self._display_frame(self.current_frame)

    # ────────────────────────────────────
    # VIDEO YÜKLEME
    # ────────────────────────────────────
    def load_video(self, video_path: str):
        """Video dosyasını yükler ve tüm kareleri belleğe alır."""
        self.stop()
        self.frames = []
        self.overlay_frames = []
        self.video_label.reset_zoom()
        self._set_speed(1.0)

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
        interval = max(1, int(1000 / self.fps / self.speed))
        self.timer.start(interval)

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
    # KARE GÖSTERİMİ (ZOOM DESTEKLİ)
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

        h, w = frame.shape[:2]
        self.video_label._frame_w = w
        self.video_label._frame_h = h

        # ── Zoom uygula (kırpma yöntemi) ──
        zoom = self.video_label.zoom_level
        if zoom > 1.0:
            # Görünür bölgenin boyutu
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)

            # Pan offset ile merkez hesapla
            center_x = int(w / 2 + self.video_label.pan_offset_x * (w - crop_w) / 2)
            center_y = int(h / 2 + self.video_label.pan_offset_y * (h - crop_h) / 2)

            # Kırpma sınırları
            x1 = max(0, center_x - crop_w // 2)
            y1 = max(0, center_y - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)

            # Sınır düzeltmesi
            if x2 - x1 < crop_w:
                x1 = max(0, x2 - crop_w)
            if y2 - y1 < crop_h:
                y1 = max(0, y2 - crop_h)

            frame = frame[y1:y2, x1:x2]

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
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h, w, ch = rgb.shape

        # QImage → QPixmap
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

        # Bilgi güncelle
        time_sec = idx / self.fps if self.fps > 0 else 0
        speed_text = f" | Hız: {self.speed}x" if self.speed != 1.0 else ""
        zoom_text = f" | Zoom: {self.video_label.zoom_level:.1f}x" if self.video_label.zoom_level > 1.0 else ""
        self.frame_info.setText(
            f"Kare: {idx + 1}/{self.total_frames} | "
            f"Zaman: {time_sec:.2f}s{speed_text}{zoom_text}"
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
