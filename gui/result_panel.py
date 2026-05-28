"""
result_panel.py — Analiz Sonuç Paneli
=======================================
Karar göstergesi, ivme-zaman grafiği, temas mesafesi grafiği
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
    QHeaderView, QFrame, QScrollArea, QGroupBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from src.config import REPORTS_DIR


class ResultPanel(QWidget):
    """
    Analiz sonuçlarını görsel olarak sunan panel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)

        # ── Karar Göstergesi ──
        self.decision_box = QGroupBox("KARAR")
        self.decision_box.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #e94560;
                border: 2px solid #16213e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
            }
        """)
        dec_layout = QVBoxLayout(self.decision_box)

        self.decision_label = QLabel("Analiz bekleniyor...")
        self.decision_label.setAlignment(Qt.AlignCenter)
        self.decision_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.decision_label.setStyleSheet("color: #8899aa; padding: 15px; background-color: #16213e; border-radius: 8px;")
        dec_layout.addWidget(self.decision_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Segoe UI", 11))
        self.confidence_label.setStyleSheet("color: #8899aa;")
        dec_layout.addWidget(self.confidence_label)

        self.reasoning_label = QLabel("")
        self.reasoning_label.setWordWrap(True)
        self.reasoning_label.setStyleSheet("""
            color: #aabbcc;
            font-size: 11px;
            padding: 5px 10px;
            background-color: #0a0a1a;
            border-radius: 4px;
        """)
        dec_layout.addWidget(self.reasoning_label)

        main_layout.addWidget(self.decision_box)

        # ── Grafikler ──
        self.graphs_box = QGroupBox("GRAFİKLER")
        self.graphs_box.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: #0f3460;
                border: 2px solid #16213e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
            }
        """)
        graph_layout = QVBoxLayout(self.graphs_box)

        # Matplotlib canvas (ivme grafiği)
        self.figure = Figure(figsize=(5, 4), dpi=80)
        self.figure.set_facecolor("#0d1117")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)
        graph_layout.addWidget(self.canvas)

        main_layout.addWidget(self.graphs_box)

        # ── Detay Tablosu ──
        self.table_box = QGroupBox("KARE BAZLI LOG")
        self.table_box.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: #0f3460;
                border: 2px solid #16213e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
            }
        """)
        table_layout = QVBoxLayout(self.table_box)

        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                gridline-color: #21262d;
                border: none;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #1f6feb;
            }
            QHeaderView::section {
                background-color: #161b22;
                color: #58a6ff;
                padding: 6px;
                border: 1px solid #21262d;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        table_layout.addWidget(self.table)

        # Rapor kaydet butonu
        self.btn_save_report = QPushButton("📄 Rapor Kaydet")
        self.btn_save_report.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:pressed {
                background-color: #1a7f37;
            }
        """)
        table_layout.addWidget(self.btn_save_report)

        main_layout.addWidget(self.table_box)

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
        indicator = decision.get("indicator", "❓")
        label_tr = decision.get("label_tr", "Belirsiz")
        color = decision.get("color", "#95A5A6")
        confidence = decision.get("confidence", 0)
        reasoning = decision.get("reasoning", "")

        self.decision_label.setText(f"{indicator}  {label_tr}")
        self.decision_label.setStyleSheet(
            f"color: {color}; padding: 15px; font-size: 26px; font-weight: 900; background-color: {color}15; border-radius: 8px; border: 1px solid {color}30;"
        )
        self.confidence_label.setText(f"Güven: {confidence:.1%}")
        self.confidence_label.setStyleSheet(f"color: {color};")
        self.reasoning_label.setText(reasoning)

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
            ax.set_facecolor("#0d1117")
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
            x_acc = np.arange(len(acc))
            ax1.plot(x_acc, acc, color="#e94560", linewidth=1.5, marker="o", markersize=3)
            ax1.fill_between(x_acc, acc, alpha=0.2, color="#e94560")
            ax1.axhline(y=0, color="#30363d", linewidth=0.5)
            ax1.set_title("İvme Profili", fontsize=10, fontweight="bold")
            ax1.set_ylabel("İvme (px/s²)", fontsize=9)
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

        for row, entry in enumerate(frame_log):
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

    def clear(self):
        """Paneli temizler."""
        self.decision_label.setText("Analiz bekleniyor...")
        self.decision_label.setStyleSheet("color: #8899aa; padding: 15px; background-color: #16213e; border-radius: 8px;")
        self.confidence_label.setText("")
        self.reasoning_label.setText("")
        self.figure.clear()
        self.canvas.draw()
        self.table.setRowCount(0)
