"""
main_window.py — Taekwondo Ghost Hit Detection: Ana Pencere
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
    QMessageBox, QStatusBar, QApplication, QComboBox,
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
            self.progress.emit(5, "Modüller yükleniyor...")
            from main import analyze_video

            self.progress.emit(10, "Video analiz ediliyor...")

            # Analiz çalıştır
            result = analyze_video(self.video_path, verbose=False,
                                   sensor_scored=self.sensor_scored,
                                   contact_ranges=self.contact_ranges)

            self.progress.emit(100, "Tamamlandı!")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """
    Taekwondo Ghost Hit Detection — Ana Uygulama Penceresi.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🥋 Taekwondo Ghost Hit Detection — Hakem Karar Destek Sistemi")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 900)

        self.current_video_path = None
        self.worker = None

        self._setup_style()
        self._setup_ui()
        self._setup_statusbar()

    def _setup_style(self):
        """Uygulama geneli koyu tema stili."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
            QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #58a6ff;
                border: 2px solid #30363d;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
            QSplitter::handle {
                background-color: #21262d;
                width: 3px;
            }
            QScrollBar:vertical {
                background-color: #0d1117;
                width: 10px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #30363d;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Başlık ──
        title_bar = QHBoxLayout()
        title_label = QLabel("🥋 TAEKWONDO GHOST HIT DETECTION")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; padding: 5px;")
        title_bar.addWidget(title_label)

        subtitle = QLabel("Bilgisayarlı Görü ve Kinematik Analiz ile Dijital Karar Destek Sistemi")
        subtitle.setStyleSheet("color: #8b949e; font-size: 11px; padding-top: 8px;")
        title_bar.addWidget(subtitle)
        title_bar.addStretch()

        main_layout.addLayout(title_bar)

        # ── Splitter (3 Panel) ──
        splitter = QSplitter(Qt.Horizontal)

        # ────────── SOL PANEL (Sadeleştirilmiş) ──────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Dosya Aç butonu
        self.btn_open = QPushButton("📂  Video Dosyası Aç")
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:pressed {
                background-color: #1a7f37;
            }
        """)
        self.btn_open.clicked.connect(self._open_file)
        left_layout.addWidget(self.btn_open)

        # Analiz butonu
        self.btn_analyze = QPushButton("🔬  Analizi Başlat")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("""
            QPushButton {
                background-color: #1f6feb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388bfd;
            }
            QPushButton:pressed {
                background-color: #1158c7;
            }
            QPushButton:disabled {
                background-color: #21262d;
                color: #484f58;
            }
        """)
        self.btn_analyze.clicked.connect(self._start_analysis)
        left_layout.addWidget(self.btn_analyze)

        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #21262d;
                border-radius: 6px;
                background-color: #161b22;
                text-align: center;
                color: #c9d1d9;
                font-size: 11px;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #1f6feb;
                border-radius: 4px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # ── Hakem girdileri: sensör durumu ve temas anı ──
        referee_box = QGroupBox("HAKEM GİRDİLERİ")
        referee_layout = QVBoxLayout(referee_box)

        sensor_label = QLabel("Sensör puan verdi mi?")
        sensor_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        referee_layout.addWidget(sensor_label)
        self.sensor_combo = QComboBox()
        # (görünen metin, analize giden değer)
        self._sensor_options = [("Bilinmiyor", None), ("Puan verdi", True), ("Puan vermedi", False)]
        for text, _ in self._sensor_options:
            self.sensor_combo.addItem(text)
        self.sensor_combo.setStyleSheet("""
            QComboBox {
                background-color: #161b22; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 6px; padding: 5px 8px; font-size: 12px;
            }
            QComboBox QAbstractItemView { background-color: #161b22; color: #c9d1d9; }
        """)
        self.sensor_combo.currentIndexChanged.connect(self._on_sensor_changed)
        referee_layout.addWidget(self.sensor_combo)

        moment_label = QLabel("Temas anı (videoyu durdurup işaretleyin):")
        moment_label.setWordWrap(True)
        moment_label.setStyleSheet("color: #8b949e; font-size: 11px; padding-top: 6px;")
        referee_layout.addWidget(moment_label)
        self.btn_mark = QPushButton("📍  Bu kareyi işaretle")
        self.btn_mark.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:disabled { color: #484f58; }
        """)
        self.btn_mark.setEnabled(False)
        self.btn_mark.clicked.connect(self._mark_current_frame)
        referee_layout.addWidget(self.btn_mark)
        self.btn_clear_marks = QPushButton("Temizle")
        self.btn_clear_marks.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:disabled { color: #484f58; }
        """)
        self.btn_clear_marks.setEnabled(False)
        self.btn_clear_marks.clicked.connect(self._clear_marks)
        referee_layout.addWidget(self.btn_clear_marks)
        self.marks_label = QLabel("İşaretli kare yok (an otomatik bulunur)")
        self.marks_label.setWordWrap(True)
        self.marks_label.setStyleSheet("color: #c9d1d9; font-size: 11px;")
        referee_layout.addWidget(self.marks_label)
        left_layout.addWidget(referee_box)
        self.marked_frames = []

        # Video bilgi paneli
        info_box = QGroupBox("VIDEO BİLGİLERİ")
        info_layout = QVBoxLayout(info_box)
        self.info_labels = {}
        for key, label in [
            ("filename", "Dosya:"),
            ("resolution", "Çözünürlük:"),
            ("fps", "FPS:"),
            ("frames", "Kare Sayısı:"),
            ("duration", "Süre:"),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8b949e; font-size: 11px; min-width: 80px;")
            val = QLabel("—")
            val.setStyleSheet("color: #c9d1d9; font-size: 11px;")
            self.info_labels[key] = val
            row.addWidget(lbl)
            row.addWidget(val, 1)
            info_layout.addLayout(row)
        left_layout.addWidget(info_box)

        # Kullanım kılavuzu
        guide_box = QGroupBox("KULLANIM")
        guide_layout = QVBoxLayout(guide_box)
        guide_text = QLabel(
            "1. Video dosyasını açın\n"
            "2. Videoyu izleyip inceleyin\n"
            "   • Zoom: Mouse tekerleği\n"
            "   • Pan: Sürükle (zoom'da)\n"
            "   • Hız: Alt kontroller\n"
            "3. (İsteğe bağlı) Sensör durumunu seçin,\n"
            "   temas anı karelerini işaretleyin\n"
            "4. Analizi başlatın\n"
            "5. Sonuçları inceleyin: grafik/tablo\n"
            "   videoyla birlikte ilerler"
        )
        guide_text.setWordWrap(True)
        guide_text.setStyleSheet("""
            color: #8b949e;
            font-size: 11px;
            padding: 5px;
            line-height: 1.4;
        """)
        guide_layout.addWidget(guide_text)
        left_layout.addWidget(guide_box)

        # Alt boşluk — sol paneli yukarı topla
        left_layout.addStretch()

        left_panel.setMaximumWidth(280)

        # ────────── ORTA PANEL (Video — Büyütülmüş) ──────────
        self.video_player = VideoPlayerWidget()

        # ────────── SAĞ PANEL ──────────
        self.result_panel = ResultPanel()
        self.result_panel.btn_save_report.clicked.connect(self._save_report)

        # Video <-> grafik/tablo eşzamanlama
        self.video_player.frame_changed.connect(self.result_panel.set_current_frame)
        self.result_panel.frame_selected.connect(self.video_player.go_to_frame)

        # Splitter'a ekle
        splitter.addWidget(left_panel)
        splitter.addWidget(self.video_player)
        splitter.addWidget(self.result_panel)

        # Oranlar: Sol %15, Orta %50, Sağ %35
        splitter.setSizes([220, 720, 500])

        main_layout.addWidget(splitter, 1)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #161b22;
                color: #8b949e;
                border-top: 1px solid #21262d;
                font-size: 11px;
                padding: 2px 10px;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır — Video yüklemek için 'Dosya Aç' butonunu kullanın")

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
            self.info_labels["filename"].setText(filename)
            self.info_labels["resolution"].setText(f"{w}x{h}")
            self.info_labels["fps"].setText(f"{fps:.0f}")
            self.info_labels["frames"].setText(str(fc))
            self.info_labels["duration"].setText(f"{dur:.2f}s")

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
        frames = sorted(set(self.marked_frames))
        ranges = []
        for f in frames:
            if ranges and f == ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], f)
            else:
                ranges.append((f, f))
        return ranges or None

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
        self.status_bar.showMessage(message)

    def _on_analysis_finished(self, result: dict):
        """Analiz tamamlandığında sonuçları gösterir."""
        self.progress_bar.setVisible(False)
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
        self.btn_analyze.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.status_bar.showMessage(f"Hata: {error_msg}")
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında hata oluştu:\n\n{error_msg}")

    # ────────────────────────────────────
    # RAPOR
    # ────────────────────────────────────
    def _save_report(self):
        """Analiz raporunu PDF olarak Masaüstüne kaydeder."""
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Bilgi", "Henüz analiz yapılmadı.")
            return

        from src.decision_engine import DecisionEngine
        engine = DecisionEngine()
        
        # Ham verileri al
        decision = self._last_result.get("decision_result", {})
        contact = self._last_result.get("contact_summary", {})
        kinematic = self._last_result.get("kinematic_result", {})

        # Masaüstü yolunu bul
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        default_name = os.path.splitext(
            os.path.basename(self.current_video_path or "report")
        )[0] + "_Rapor.pdf"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "PDF Raporunu Kaydet",
            os.path.join(desktop_path, default_name),
            "PDF Dosyası (*.pdf);;Tüm Dosyalar (*)",
        )

        if file_path:
            try:
                from fpdf import FPDF
                from datetime import datetime
                
                class PDFReport(FPDF):
                    def header(self):
                        # Logo/Başlık Arka Planı
                        self.set_fill_color(22, 33, 62)
                        self.rect(0, 0, 210, 30, 'F')
                        
                        self.set_font('Arial', 'B', 18)
                        self.set_text_color(255, 255, 255)
                        self.cell(0, 15, 'TAEKWONDO GHOST HIT DETECTION', 0, 1, 'C')
                        
                        self.set_font('Arial', '', 10)
                        self.set_text_color(200, 200, 200)
                        self.cell(0, 5, 'Resmi Karar Destek Sistemi Analiz Raporu', 0, 1, 'C')
                        self.ln(15)
                        
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.set_text_color(128, 128, 128)
                        self.cell(0, 10, f'Sayfa {self.page_no()} | Tarih: {datetime.now().strftime("%d-%m-%Y %H:%M")}', 0, 0, 'C')

                pdf = PDFReport()
                pdf.add_page()
                
                def safe_text(text):
                    tr_map = {'ı':'i', 'i':'i', 'ş':'s', 'Ş':'S', 'ğ':'g', 'Ğ':'G', 'ü':'u', 'Ü':'U', 'ö':'o', 'Ö':'O', 'ç':'c', 'Ç':'C'}
                    clean = "".join([tr_map.get(c, str(c)) for c in str(text)])
                    return clean.encode('latin-1', 'ignore').decode('latin-1')

                # KARAR KUTUSU
                decision_label = decision.get('label_tr', 'Bilinmiyor')
                confidence = decision.get('confidence', 0.0) * 100
                
                # Karar rengi, arayüzdeki ile ayni olsun diye karar motorundan alinir
                hex_color = decision.get('color', '#95A5A6').lstrip('#')
                r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                pdf.set_fill_color(int(r + (255 - r) * 0.85), int(g + (255 - g) * 0.85), int(b + (255 - b) * 0.85))
                pdf.set_text_color(int(r * 0.75), int(g * 0.75), int(b * 0.75))
                
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 15, safe_text(f"  Nihai Karar: {decision_label} (Guven Orani: %{confidence:.1f})"), 0, 1, 'L', fill=True)
                pdf.ln(5)

                # GEREKÇE (REASONING)
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(33, 37, 41)
                pdf.cell(0, 10, "Gerekce:", 0, 1)
                
                pdf.set_font("Arial", '', 11)
                pdf.set_text_color(73, 80, 87)
                pdf.multi_cell(0, 6, safe_text(decision.get('reasoning', '')))
                pdf.ln(10)
                
                # ÇİZGİ
                pdf.set_draw_color(206, 212, 218)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(10)

                # DETAYLAR BÖLÜMÜ (2 SÜTUN)
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(33, 37, 41)
                pdf.cell(0, 10, "Teknik Analiz Detaylari:", 0, 1)
                pdf.ln(2)
                
                # Temas ve Kinematik tabloları yan yana
                y_start = pdf.get_y()
                
                # Sol Sütun (Temas Analizi)
                pdf.set_fill_color(248, 249, 250)
                pdf.rect(10, y_start, 90, 60, 'F')
                
                pdf.set_font("Arial", 'B', 10)
                pdf.set_text_color(22, 33, 62)
                pdf.set_xy(12, y_start + 2)
                pdf.cell(86, 8, "Temas Verileri", 0, 1)
                
                pdf.set_font("Arial", '', 10)
                pdf.set_text_color(73, 80, 87)
                
                def add_row(x, y, k, v):
                    pdf.set_xy(x, y)
                    pdf.set_font("Arial", 'B', 9)
                    pdf.cell(40, 8, safe_text(k))
                    pdf.set_font("Arial", '', 9)
                    pdf.cell(40, 8, safe_text(v))
                
                add_row(12, y_start + 12, "Toplam Kare:", f"{contact.get('total_frames', 0)}")
                add_row(12, y_start + 20, "Temasli Kare:", f"{contact.get('contact_frames', 0)}")
                add_row(12, y_start + 28, "Yakin Gecis:", f"{contact.get('near_miss_frames', 0)}")
                add_row(12, y_start + 36, "Maks Cakisma:", f"{contact.get('max_overlap', 0)} px")
                min_dist = contact.get('min_distance', float('inf'))
                add_row(12, y_start + 44, "Min Mesafe:", f"{min_dist:.1f} px" if min_dist < 1e6 else "N/A")
                
                # Sağ Sütun (Kinematik Analiz)
                pdf.set_fill_color(248, 249, 250)
                pdf.rect(110, y_start, 90, 60, 'F')
                
                pdf.set_font("Arial", 'B', 10)
                pdf.set_text_color(22, 33, 62)
                pdf.set_xy(112, y_start + 2)
                pdf.cell(86, 8, "Kinematik Verileri", 0, 1)
                
                add_row(112, y_start + 12, "Hareket Tipi:", f"{kinematic.get('type', 'N/A').upper()}")
                add_row(112, y_start + 20, "Maks Ivme:", f"{kinematic.get('max_acceleration', 0):.1f} px/s^2")
                add_row(112, y_start + 28, "Ivme Tepe Karesi:", f"{kinematic.get('peak_frame', 'N/A')}")
                add_row(112, y_start + 36, "Varyasyon (CV):", f"{kinematic.get('acceleration_cv', 0):.2f}")
                add_row(112, y_start + 44, "Net Savrulma:", f"{kinematic.get('net_displacement', 0):.1f} px")
                
                pdf.output(file_path)
                self.status_bar.showMessage(f"PDF Raporu kaydedildi: {file_path}")
                QMessageBox.information(self, "Başarılı", f"Resmi PDF Raporu Başarıyla Oluşturuldu!\n\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"PDF oluşturulurken hata meydana geldi:\n{str(e)}")
