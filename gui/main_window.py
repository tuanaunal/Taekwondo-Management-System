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
    QMessageBox, QStatusBar, QApplication,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QColor

# Proje kök dizinini Python yoluna ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.video_player import VideoPlayerWidget
from gui.result_panel import ResultPanel
from src.preprocessor import VideoPreprocessor
from src.segmentor import EquipmentSegmentor
from src.config import GRAPHS_DIR, REPORTS_DIR


class AnalysisWorker(QThread):
    """
    Analiz işlemini arka planda çalıştıran thread.
    GUI donmasını önler.
    """
    progress = pyqtSignal(int, str)     # (yüzde, mesaj)
    finished = pyqtSignal(dict)         # analiz sonucu
    error = pyqtSignal(str)             # hata mesajı

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.progress.emit(5, "Modüller yükleniyor...")
            from main import analyze_video

            self.progress.emit(10, "Video analiz ediliyor...")

            # Analiz çalıştır
            result = analyze_video(self.video_path, verbose=False)

            self.progress.emit(80, "Overlay kareleri oluşturuluyor...")

            # Overlay kareleri oluştur
            overlay_frames = self._generate_overlays(self.video_path)
            result["overlay_frames"] = overlay_frames

            self.progress.emit(100, "Tamamlandı!")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _generate_overlays(self, video_path: str) -> list:
        """Segmentasyon overlay karelerini oluşturur."""
        preprocessor = VideoPreprocessor()
        segmentor = EquipmentSegmentor()
        overlays = []

        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = preprocessor.process_frame(frame)
            seg_results = segmentor.segment_frame(processed)
            overlay = segmentor.draw_segmentation(frame, seg_results, alpha=0.5)
            overlays.append(overlay)
        cap.release()

        return overlays


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
            "3. Analizi başlatın\n"
            "4. Sonuçları inceleyin"
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

        self.worker = AnalysisWorker(self.current_video_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.status_bar.showMessage(message)

    def _on_analysis_finished(self, result: dict):
        """Analiz tamamlandığında sonuçları gösterir."""
        self.progress_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_open.setEnabled(True)

        # Overlay karelerini video oynatıcıya aktar
        overlay_frames = result.pop("overlay_frames", [])
        if overlay_frames:
            self.video_player.set_overlay_frames(overlay_frames)

        # Sonuçları panelde göster
        self.result_panel.display_results(result)

        # Analiz sonucunu sakla
        self._last_result = result

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
        \"\"\"Analiz raporunu PDF olarak Masaüstüne kaydeder.\"\"\"
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Bilgi", "Henüz analiz yapılmadı.")
            return

        from src.decision_engine import DecisionEngine
        engine = DecisionEngine()
        report_text = engine.format_report(self._last_result["decision_result"])

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
                
                class PDFReport(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 15)
                        self.set_text_color(233, 69, 96) # #e94560 renk
                        self.cell(0, 10, 'TAEKWONDO GHOST HIT - ANALIZ RAPORU', 0, 1, 'C')
                        self.set_draw_color(233, 69, 96)
                        self.line(10, 20, 200, 20)
                        self.ln(10)
                        
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.set_text_color(128, 128, 128)
                        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

                pdf = PDFReport()
                pdf.add_page()
                pdf.set_font("Arial", size=11)
                pdf.set_text_color(40, 40, 40)
                
                # Türkçe karakterleri İngilizce eşdeğerlerine çevir (FPDF standart font uyumluluğu için)
                tr_map = {'ı':'i', 'i':'i', 'ş':'s', 'Ş':'S', 'ğ':'g', 'Ğ':'G', 'ü':'u', 'Ü':'U', 'ö':'o', 'Ö':'O', 'ç':'c', 'Ç':'C'}
                clean_text = "".join([tr_map.get(c, c) for c in report_text])
                
                for line in clean_text.split('\\n'):
                    if "---" in line or "===" in line:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.set_text_color(31, 111, 235) # Mavi başlık
                        pdf.cell(0, 10, txt=line, ln=1)
                        pdf.set_font("Arial", size=11)
                        pdf.set_text_color(40, 40, 40)
                    else:
                        pdf.multi_cell(0, 8, txt=line)
                
                pdf.output(file_path)
                self.status_bar.showMessage(f"PDF Raporu kaydedildi: {file_path}")
                QMessageBox.information(self, "Başarılı", f"PDF Raporu Başarıyla Oluşturuldu!\\n\\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"PDF oluşturulurken hata meydana geldi:\\n{str(e)}")
