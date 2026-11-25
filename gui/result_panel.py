"""
result_panel.py — Analiz Sonuç Paneli
=======================================
Karar kartı, ivme-zaman grafiği, temas mesafesi grafiği
ve detaylı log tablosu içeren sonuç paneli.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QFrame, QScrollArea, QGroupBox, QSizePolicy, QTabWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from src.config import REPORTS_DIR
from gui import theme
from gui.decision_card import DecisionCard


class ResultPanel(QWidget):
    """
    Analiz sonuçlarını görsel olarak sunan panel.

    Video ile eşzamanlı çalışır: set_current_frame() grafiklerdeki dikey kare çizgisini
    ve tablodaki seçili satırı günceller; tablodaki bir satıra tıklanınca frame_selected
    sinyaliyle video o kareye gider.
    """

    frame_selected = pyqtSignal(int)  # Kullanıcı tablodan bir kare seçti

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_lines = []   # Grafiklerdeki dikey "şu anki kare" çizgileri
        self._row_for_frame = {}  # kare numarası -> tablo satırı
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # ── Karar kartı (her zaman tam boyutunda görünür) ──
        self.decision_card = DecisionCard()
        self.decision_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        main_layout.addWidget(self.decision_card, 0)

        # ── Ayrıntılar: grafikler ve kare tablosu sekmelerde (dikey yer kazanmak için) ──
        self.tabs = QTabWidget()

        graphs_page = QWidget()
        graph_layout = QVBoxLayout(graphs_page)
        graph_layout.setContentsMargins(6, 6, 6, 6)
        self.figure = Figure(figsize=(5, 4), dpi=80)
        self.figure.set_facecolor(theme.SURFACE)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(160)
        graph_layout.addWidget(self.canvas)
        self.tabs.addTab(graphs_page, "Grafikler")

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(6, 6, 6, 6)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellClicked.connect(self._on_table_clicked)
        table_layout.addWidget(self.table)
        self.tabs.addTab(table_page, "Kare tablosu")

        main_layout.addWidget(self.tabs, 1)

        # Rapor kaydet butonu
        self.btn_save_report = QPushButton("📄  PDF Rapor Kaydet")
        self.btn_save_report.setProperty("variant", "secondary")
        self.btn_save_report.setEnabled(False)
        main_layout.addWidget(self.btn_save_report)

    # ────────────────────────────────────
    # SONUÇLARI GÖSTER
    # ────────────────────────────────────
    def display_results(self, analysis_result: dict):
        """
        Analiz sonuçlarını panelde gösterir.

        Parameters
        ----------
        analysis_result : dict
            main.analyze_video() çıktısı.
        """
        decision = analysis_result.get("decision_result", {})
        contact = analysis_result.get("contact_summary", {})
        kinematic = analysis_result.get("kinematic_result", {})
        frame_log = analysis_result.get("frame_log", [])

        # ── Karar ──
        self.decision_card.display(decision)
        self.btn_save_report.setEnabled(True)

        # ── Grafik ──
        self._plot_analysis(contact, kinematic)

        # ── Tablo ──
        self._populate_table(frame_log)

    def _plot_analysis(self, contact: dict, kinematic: dict):
        """İvme ve temas mesafesi grafiklerini çizer."""
        self.figure.clear()

        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        # Stil
        for ax in [ax1, ax2]:
            ax.set_facecolor(theme.SURFACE)
            ax.tick_params(colors="#8b949e", labelsize=8)
            ax.spines["bottom"].set_color("#30363d")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#30363d")
            ax.yaxis.label.set_color("#c9d1d9")
            ax.xaxis.label.set_color("#c9d1d9")
            ax.title.set_color("#c9d1d9")

        # ── İvme Grafiği ──
        acc = kinematic.get("acceleration_profile", np.array([]))
        if hasattr(acc, '__len__') and len(acc) > 0:
            # Yatay eksen: videodaki kare numarası (video ve tabloyla aynı eksen)
            x_acc = kinematic.get("profile_start_frame", 0) + np.arange(len(acc))
            ax1.plot(x_acc, acc, color="#e94560", linewidth=1.5, marker="o", markersize=3)
            ax1.fill_between(x_acc, acc, alpha=0.2, color="#e94560")
            ax1.axhline(y=0, color="#30363d", linewidth=0.5)
            ax1.set_title("İvme Profili", fontsize=10, fontweight="bold")
            ax1.set_ylabel("İvme (px/s²)", fontsize=9)
            ax1.set_xlabel("Kare", fontsize=9)
        else:
            ax1.text(0.5, 0.5, "İvme verisi yok",
                     ha="center", va="center", color="#8b949e",
                     transform=ax1.transAxes)

        # ── Temas Mesafesi Grafiği ──
        distances = contact.get("distance_profile", [])
        overlaps = contact.get("overlap_profile", [])

        if distances:
            x = np.arange(len(distances))
            # inf değerleri temizle
            clean_dist = [d if d < 1e6 else np.nan for d in distances]
            ax2.plot(x, clean_dist, color="#58a6ff", linewidth=1.5, marker="s", markersize=3, label="Kontur Mesafesi")
            ax2.fill_between(x, clean_dist, alpha=0.15, color="#58a6ff")

            if overlaps:
                ax2_twin = ax2.twinx()
                ax2_twin.bar(x, overlaps, alpha=0.4, color="#2ea043", label="Çakışma (px)")
                ax2_twin.set_ylabel("Çakışma (px)", fontsize=9, color="#2ea043")
                ax2_twin.tick_params(axis="y", colors="#2ea043", labelsize=8)
                ax2_twin.spines["right"].set_color("#2ea043")

            ax2.set_title("Temas Analizi", fontsize=10, fontweight="bold")
            ax2.set_ylabel("Mesafe (px)", fontsize=9)
            ax2.set_xlabel("Kare", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "Temas verisi yok",
                     ha="center", va="center", color="#8b949e",
                     transform=ax2.transAxes)

        # Videoyla eşzamanlı ilerleyen "şu anki kare" çizgisi
        self._frame_lines = [ax.axvline(0, color="#f1c40f", linewidth=1.5, alpha=0.9)
                             for ax in (ax1, ax2)]
        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()

    def _populate_table(self, frame_log: list):
        """Kare bazlı log tablosunu doldurur."""
        if not frame_log:
            self.table.setRowCount(0)
            return

        columns = ["Kare", "Zaman(s)", "Temas", "Çakışma(px)", "Merkez Mes.", "Kontur Mes."]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(frame_log))

        self._row_for_frame = {}
        for row, entry in enumerate(frame_log):
            self._row_for_frame[int(entry.get("frame", row))] = row
            items = [
                str(entry.get("frame", "")),
                f"{entry.get('time_sec', 0):.3f}",
                entry.get("contact_type", "N/A"),
                str(entry.get("overlap_px", 0)),
                f"{entry.get('centroid_dist', 0):.1f}" if entry.get('centroid_dist', float('inf')) < 1e6 else "N/A",
                f"{entry.get('contour_dist', 0):.1f}" if entry.get('contour_dist', float('inf')) < 1e6 else "N/A",
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)

                # Renk kodlama
                contact_type = entry.get("contact_type", "")
                if contact_type == "overlap":
                    item.setBackground(QColor(46, 160, 67, 40))
                elif contact_type == "near_miss":
                    item.setBackground(QColor(210, 153, 34, 40))

                self.table.setItem(row, col, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # ────────────────────────────────────
    # VİDEO İLE EŞZAMANLAMA
    # ────────────────────────────────────
    def set_current_frame(self, idx: int):
        """Videoda gösterilen kareyi grafiklerde (dikey çizgi) ve tabloda (seçili satır) gösterir."""
        if self._frame_lines:
            for line in self._frame_lines:
                line.set_xdata([idx, idx])
            self.canvas.draw_idle()
        row = self._row_for_frame.get(int(idx))
        if row is not None and row < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))
            self.table.blockSignals(False)

    def _on_table_clicked(self, row: int, _col: int):
        item = self.table.item(row, 0)
        if item is not None and item.text().isdigit():
            self.frame_selected.emit(int(item.text()))

    def clear(self):
        """Paneli temizler."""
        self._frame_lines = []
        self._row_for_frame = {}
        self.decision_card.clear()
        self.btn_save_report.setEnabled(False)
        self.figure.clear()
        self.canvas.draw()
        self.table.setRowCount(0)
