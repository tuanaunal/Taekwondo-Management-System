"""
main_window.py — Hakem Karar Destek Sistemi: Ana Pencere
=============================================================
Profesyonel masaüstü hakem kontrol paneli.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  BAŞLIK BAR                                         │
  ├──────────────┬──────────────────────┬───────────────┤
  │  Sol Panel   │   Orta Panel         │  Sağ Panel    │
  │  - Dosya Aç  │   - Video Oynatıcı   │  - Karar      │
  │  - Analiz    │   - Zoom/Hız         │  - Grafikler  │
  │  - Bilgi     │                      │  - Log Tablo  │
  │              │                      │  - Rapor      │
  └──────────────┴──────────────────────┴───────────────┘
"""

import os
import sys
import cv2

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QListWidget,
    QListWidgetItem, QSplitter, QGroupBox, QProgressBar,
    QMessageBox, QStatusBar, QApplication, QComboBox, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QColor

# Proje kök dizinini Python yoluna ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.video_player import VideoPlayerWidget
from gui.result_panel import ResultPanel
from src.config import GRAPHS_DIR, REPORTS_DIR
from gui import theme


class AnalysisWorker(QThread):
    """
    Analiz işlemini arka planda çalıştıran thread.
    GUI donmasını önler.
    """
    progress = pyqtSignal(int, str)     # (yüzde, mesaj)
    finished = pyqtSignal(dict)         # analiz sonucu
    error = pyqtSignal(str)             # hata mesajı

    def __init__(self, video_path: str, sensor_scored=None, contact_ranges=None):
        super().__init__()
        self.video_path = video_path
        self.sensor_scored = sensor_scored      # True / False / None (bilinmiyor)
        self.contact_ranges = contact_ranges    # Hakemin işaretlediği temas anları

    def run(self):
        try:
            self.progress.emit(1, "Modüller yükleniyor...")
            from main import analyze_video

            self.progress.emit(3, "Video açılıyor...")

            # Analiz çalıştır
            result = analyze_video(self.video_path, verbose=False,
                                   sensor_scored=self.sensor_scored,
                                   contact_ranges=self.contact_ranges,
                                   progress_callback=self.progress.emit)

            self.progress.emit(100, "Tamamlandı!")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """
    Hakem Karar Destek Sistemi — Ana Uygulama Penceresi.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taekwondo Hakem Karar Destek Sistemi")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 900)

        self.current_video_path = None
        self.worker = None

        self.setStyleSheet(theme.APP_STYLESHEET)
        self._setup_ui()
        self._setup_statusbar()

    # ────────────────────────────────────
    # ARAYÜZ
    # ────────────────────────────────────
    @staticmethod
    def _card(title: str, step: str = None, hint: str = None):
        """Başlıklı kart çerçevesi döndürür: (kart, içerik düzeni)."""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        if step:
            badge = QLabel(step)
            badge.setProperty("role", "step")
            badge.setAlignment(Qt.AlignCenter)
            head.addWidget(badge)
        title_lbl = QLabel(title)
        title_lbl.setProperty("role", "section")
        head.addWidget(title_lbl)
        head.addStretch()
        if hint:
            hint_lbl = QLabel(hint)
            hint_lbl.setProperty("role", "muted")
            head.addWidget(hint_lbl)
        layout.addLayout(head)
        return card, layout

    def _build_header(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 4)
        bar.setSpacing(12)

        emblem = QLabel("TKD")
        emblem.setAlignment(Qt.AlignCenter)
        emblem.setFixedSize(44, 44)
        emblem.setStyleSheet(
            f"background-color: {theme.rgba(theme.BRAND_RED, 0.18)}; border: 1px solid "
            f"{theme.rgba(theme.BRAND_RED, 0.5)}; border-radius: 12px; font-size: 13px; font-weight: 800;"
            f" color: {theme.BRAND_RED}; letter-spacing: 1px;")
        bar.addWidget(emblem)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        overline = QLabel("TAEKWONDO  ·  BİLGİSAYARLI GÖRÜ")
        overline.setStyleSheet(f"color: {theme.BRAND_RED}; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
        title = QLabel("Hakem Karar Destek Sistemi")
        title.setStyleSheet(f"color: {theme.TEXT}; font-size: 22px; font-weight: 800;")
        subtitle = QLabel("Kafa vuruşlarında elektronik puanlama hatalarının (Ghost Hit, kaçırılan vuruş) görüntüden tespiti")
        subtitle.setProperty("role", "muted")
        titles.addWidget(overline)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        bar.addLayout(titles)
        bar.addStretch()

        # Dürüst doğruluk rozeti: model dosyasındaki doğrulama sonucu
        acc = self._honest_accuracy()
        if acc is not None:
            chip = QLabel(f"Görülmemiş kliplerde doğruluk  %{acc * 100:.0f}")
            chip.setToolTip("Her klip tüm kopyalarıyla bir kez dışarıda tutularak ölçüldü "
                            "(birini dışarıda bırakarak doğrulama, 20 klip). Ayrıntı: SONUC_RAPORU.md")
            chip.setStyleSheet(
                f"background-color: {theme.SURFACE}; border: 1px solid {theme.BORDER}; border-radius: 14px;"
                f" padding: 6px 14px; color: {theme.TEXT_2}; font-size: 11px; font-weight: 600;")
            bar.addWidget(chip, 0, Qt.AlignVCenter)
        return bar

    @staticmethod
    def _honest_accuracy():
        try:
            from src.decision_engine import load_model
            model = load_model()
            return model.get("dogrulama", {}).get("orijinal_20_dogruluk") if model else None
        except Exception:
            return None

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── 1. Video ──
        card, c = self._card("VİDEO", step="1")
        self.btn_open = QPushButton("📂  Video Aç")
        self.btn_open.setProperty("variant", "secondary")
        self.btn_open.clicked.connect(self._open_file)
        c.addWidget(self.btn_open)
        self.file_label = QLabel("Henüz video seçilmedi")
        self.file_label.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")
        self.file_label.setWordWrap(True)
        c.addWidget(self.file_label)
        self.meta_label = QLabel("")
        self.meta_label.setProperty("role", "muted")
        self.meta_label.setWordWrap(True)
        c.addWidget(self.meta_label)
        layout.addWidget(card)

        # ── 2. Hakem girdileri ──
        card, c = self._card("HAKEM GİRDİLERİ", step="2", hint="isteğe bağlı")
        sensor_label = QLabel("Sensör puan verdi mi?")
        sensor_label.setProperty("role", "muted")
        c.addWidget(sensor_label)
        self.sensor_combo = QComboBox()
        # (görünen metin, analize giden değer)
        self._sensor_options = [("Bilinmiyor", None), ("Puan verdi", True), ("Puan vermedi", False)]
        for text, _ in self._sensor_options:
            self.sensor_combo.addItem(text)
        self.sensor_combo.currentIndexChanged.connect(self._on_sensor_changed)
        c.addWidget(self.sensor_combo)

        moment_label = QLabel("Temas anı — videoyu durdurup işaretleyin:")
        moment_label.setProperty("role", "muted")
        moment_label.setWordWrap(True)
        c.addSpacing(4)
        c.addWidget(moment_label)
        mark_row = QHBoxLayout()
        mark_row.setSpacing(6)
        self.btn_mark = QPushButton("📍  İşaretle")
        self.btn_mark.setEnabled(False)
        self.btn_mark.clicked.connect(self._mark_current_frame)
        self.btn_clear_marks = QPushButton("Temizle")
        self.btn_clear_marks.setEnabled(False)
        self.btn_clear_marks.clicked.connect(self._clear_marks)
        mark_row.addWidget(self.btn_mark, 1)
        mark_row.addWidget(self.btn_clear_marks)
        c.addLayout(mark_row)
        self.marks_label = QLabel("İşaretli kare yok (an otomatik bulunur)")
        self.marks_label.setProperty("role", "muted")
        self.marks_label.setWordWrap(True)
        c.addWidget(self.marks_label)
        layout.addWidget(card)
        self.marked_frames = []

        # ── 3. Analiz ──
        card, c = self._card("ANALİZ", step="3")
        self.btn_analyze = QPushButton("▶  Analizi Başlat")
        self.btn_analyze.setProperty("variant", "primary")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._start_analysis)
        c.addWidget(self.btn_analyze)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        c.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.setProperty("role", "muted")
        self.progress_label.setVisible(False)
        c.addWidget(self.progress_label)
        layout.addWidget(card)

        layout.addStretch()

        tips = QLabel("İpucu: fare tekerleğiyle yakınlaştırın, sürükleyerek kaydırın. "
                      "Grafikte veya tabloda bir kareye tıklayınca video o kareye gider.")
        tips.setWordWrap(True)
        tips.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px; padding: 0 4px;")
        layout.addWidget(tips)

        panel.setMinimumWidth(250)
        panel.setMaximumWidth(300)
        return panel

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 10, 12, 8)
        main_layout.setSpacing(10)

        main_layout.addLayout(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)

        left_panel = self._build_left_panel()

        # ── Orta: video ──
        video_card = QFrame()
        video_card.setObjectName("card")
        vc = QVBoxLayout(video_card)
        vc.setContentsMargins(10, 10, 10, 10)
        self.video_player = VideoPlayerWidget()
        vc.addWidget(self.video_player)

        # ── Sağ: sonuçlar ──
        self.result_panel = ResultPanel()
        self.result_panel.btn_save_report.clicked.connect(self._save_report)

        # Video <-> grafik/tablo eşzamanlama
        self.video_player.frame_changed.connect(self.result_panel.set_current_frame)
        self.result_panel.frame_selected.connect(self.video_player.go_to_frame)

        splitter.addWidget(left_panel)
        splitter.addWidget(video_card)
        splitter.addWidget(self.result_panel)
        splitter.setSizes([270, 700, 470])
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır — başlamak için bir video açın")

    # ────────────────────────────────────
    # DOSYA İŞLEMLERİ
    # ────────────────────────────────────
    def _open_file(self):
        """Video dosyası açma diyalogu."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Video Dosyası Seç",
            os.path.join(project_root, "Dataset"),
            "Video Dosyaları (*.mp4 *.avi *.mov *.mkv);;Tüm Dosyalar (*)",
        )
        if file_path:
            self._load_video(file_path)

    def _load_video(self, video_path: str):
        """Videoyu oynatıcıya yükler."""
        self.current_video_path = video_path
        success = self.video_player.load_video(video_path)

        if success:
            # Video bilgilerini güncelle
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = fc / fps if fps > 0 else 0
            cap.release()

            filename = os.path.basename(video_path)
            self.file_label.setText(filename)
            self.meta_label.setText(f"{w}×{h}  ·  {fps:.0f} FPS  ·  {fc} kare  ·  {dur:.2f} sn")

            self.btn_analyze.setEnabled(True)
            self.btn_mark.setEnabled(True)
            self._clear_marks()
            self.result_panel.clear()
            self.status_bar.showMessage(f"Video yüklendi: {filename}")
        else:
            QMessageBox.warning(self, "Hata", f"Video açılamadı:\n{video_path}")

    # ────────────────────────────────────
    # ANALİZ
    # ────────────────────────────────────
    def _start_analysis(self):
        """Analizi arka plan thread'inde başlatır."""
        if self.current_video_path is None:
            return

        self.btn_analyze.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Analiz başlıyor...")
        self.status_bar.showMessage("Analiz devam ediyor...")

        self.worker = AnalysisWorker(
            self.current_video_path,
            sensor_scored=self._sensor_value(),
            contact_ranges=self._marked_ranges(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    # ────────────────────────────────────
    # HAKEM GİRDİLERİ
    # ────────────────────────────────────
    def _sensor_value(self):
        return self._sensor_options[self.sensor_combo.currentIndex()][1]

    def _marked_ranges(self):
        """İşaretli kareleri ardışık aralıklara çevirir, ör. [10, 11, 12, 20] -> [(10, 12), (20, 20)]."""
        from src.contact_moment import frames_to_ranges
        return frames_to_ranges(self.marked_frames) or None

    def _update_marks_label(self):
        ranges = self._marked_ranges()
        if not ranges:
            self.marks_label.setText("İşaretli kare yok (an otomatik bulunur)")
        else:
            parts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
            self.marks_label.setText("İşaretli kareler: " + ", ".join(parts) +
                                     "\n(Analizi yeniden başlatın)")
        self.btn_clear_marks.setEnabled(bool(ranges))

    def _mark_current_frame(self):
        self.video_player.pause()
        self.marked_frames.append(self.video_player.current_frame)
        self._update_marks_label()

    def _clear_marks(self):
        self.marked_frames = []
        self._update_marks_label()

    def _on_sensor_changed(self, _index: int):
        """Analiz yapılmışsa sensör durumu değişince kararı yeniden analiz etmeden günceller."""
        if not hasattr(self, "_last_result"):
            return
        from src.decision_engine import DecisionEngine
        r = self._last_result
        r["decision_result"] = DecisionEngine().make_decision(
            r["contact_summary"], r["kinematic_result"],
            head_response=r.get("head_response"), sensor_scored=self._sensor_value())
        self.result_panel.display_results(r)
        d = r["decision_result"]
        self.status_bar.showMessage(f"Karar güncellendi: {d['label_tr']}")

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"%{percent}  ·  {message}")
        self.status_bar.showMessage(message)

    def _on_analysis_finished(self, result: dict):
        """Analiz tamamlandığında sonuçları gösterir."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_open.setEnabled(True)

        # İşaretler: analizin kare verileri oynatıcıya verilir, o anki kareye çizilir
        # (eskiden video YOLO ile ikinci kez işlenip tüm işaretli kareler bellekte tutuluyordu)
        moment = result.get("contact_moment", {})
        self.video_player.set_frame_features(
            result.get("frame_features", []),
            result.get("video_info", {}).get("scale_factor", 1.0),
            highlight_frame=moment.get("frame"),
        )
        self.video_player.set_overlay(True)  # Analizden sonra baş/ayak işaretleri açık gelir

        # Sonuçları panelde göster
        self.result_panel.display_results(result)

        # Analiz sonucunu sakla
        self._last_result = result

        moment = result.get("contact_moment", {})
        if moment.get("frame") is not None:
            self.status_bar.showMessage(
                f"Temas anı: kare {moment['frame']} ({moment.get('method_tr', '')})")

        decision = result.get("decision_result", {})
        label_tr = decision.get("label_tr", "")
        confidence = decision.get("confidence", 0)

        self.status_bar.showMessage(
            f"Analiz tamamlandı! Sonuç: {label_tr} (güven: {confidence:.1%})"
        )

    def _on_analysis_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.status_bar.showMessage(f"Hata: {error_msg}")
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında hata oluştu:\n\n{error_msg}")

    # ────────────────────────────────────
    # RAPOR
    # ────────────────────────────────────
    def _save_report(self):
        """Analiz raporunu PDF olarak kaydeder (varsayılan konum: Masaüstü)."""
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Bilgi", "Henüz analiz yapılmadı.")
            return

        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        default_name = os.path.splitext(
            os.path.basename(self.current_video_path or "rapor")
        )[0] + "_Rapor.pdf"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "PDF Raporunu Kaydet",
            os.path.join(desktop_path, default_name),
            "PDF Dosyası (*.pdf);;Tüm Dosyalar (*)",
        )
        if not file_path:
            return
        try:
            from src.pdf_report import create_pdf_report
            create_pdf_report(self._last_result, self.current_video_path, file_path)
            self.status_bar.showMessage(f"PDF raporu kaydedildi: {file_path}")
            QMessageBox.information(self, "Başarılı", f"PDF raporu oluşturuldu:\n\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"PDF oluşturulurken hata meydana geldi:\n{str(e)}")
