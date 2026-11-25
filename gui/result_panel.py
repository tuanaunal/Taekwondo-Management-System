"""
result_panel.py — Analiz Sonuç Paneli
=======================================
Karar kartı, baş dönüşü ve ayak-baş mesafesi grafikleri ve kare kare durum
tablosu içeren sonuç paneli.
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

from src.config import REPORTS_DIR, CONTACT_MARGIN_RADII, NEAR_MISS_MARGIN_RADII
from src.decision_engine import CONTACT_GAP_CAP
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
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        graph_layout.addWidget(self.canvas)
        self.tabs.addTab(graphs_page, "Grafikler")

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(6, 6, 6, 6)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellClicked.connect(self._on_table_clicked)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setShowGrid(False)
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
        frame_log = analysis_result.get("frame_log", [])
        head = analysis_result.get("head_response", {}) or {}
        moment = analysis_result.get("contact_moment", {}) or {}
        ranges = analysis_result.get("contact_ranges") or []

        # ── Karar ──
        self.decision_card.display(decision)
        self.btn_save_report.setEnabled(True)

        rotation = {c: (head.get(c, {}) or {}).get("rotation_series") or [] for c in ("red", "blue")}

        # ── Grafik ──
        self._plot_analysis(frame_log, rotation, moment, ranges)

        # ── Tablo ──
        self._populate_table(frame_log, rotation, moment.get("frame"))

    # ────────────────────────────────────
    # GRAFİKLER
    # ────────────────────────────────────
    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor(theme.SURFACE)
        ax.tick_params(colors=theme.TEXT_2, labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme.BORDER)
        ax.grid(axis="y", color=theme.BORDER, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    def _plot_analysis(self, frame_log: list, rotation: dict, moment: dict, ranges: list):
        """
        Kararın dayandığı ölçümleri kare kare çizer:
          1. Baş dönüşü (kırmızı ve mavi sporcu) — modelin en etkili ölçümü
          2. Ayak-baş mesafesi (baş yarıçapı) — temas / yakın geçiş bölgeleriyle
        Temas anı ve hakemin işaretlediği kareler iki grafikte de gösterilir.
        """
        self.figure.clear()
        n = len(frame_log)
        if n == 0:
            self.canvas.draw()
            return
        x = np.arange(n)
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2, sharex=ax1)
        for ax in (ax1, ax2):
            self._style_axes(ax)
            # Hakemin işaretlediği kareler
            for a, b in ranges:
                ax.axvspan(a - 0.5, b + 0.5, color=theme.ACCENT, alpha=0.18, linewidth=0)
            # Temas anı
            if moment.get("frame") is not None:
                ax.axvline(moment["frame"], color=theme.TEXT_2, linewidth=1, linestyle=(0, (3, 3)))

        # ── 1. Baş dönüşü ──
        drawn = False
        for color, line_color, name in (("red", theme.BRAND_RED, "Kırmızı"), ("blue", theme.BRAND_BLUE, "Mavi")):
            series = rotation.get(color) or []
            if not series:
                continue
            y = np.array([np.nan if v is None else v for v in series[:n]], dtype=float)
            if np.isnan(y).all():
                continue
            ax1.plot(np.arange(len(y)), y, color=line_color, linewidth=2, label=name)
            drawn = True
        ax1.set_title("Baş dönüşü", fontsize=10, color=theme.TEXT, loc="left", fontweight="bold")
        ax1.set_ylabel("baş yarıçapı", fontsize=8, color=theme.TEXT_2)
        if drawn:
            leg = ax1.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.98),
                             ncol=2, labelcolor=theme.TEXT_2, borderaxespad=0)
            for h in leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles:
                h.set_linewidth(3)
            ax1.set_ylim(bottom=0)
        else:
            ax1.text(0.5, 0.5, "Baş dönüşü ölçülemedi", ha="center", va="center",
                     color=theme.TEXT_2, transform=ax1.transAxes, fontsize=9)
        ax1.tick_params(labelbottom=False)

        # ── 2. Ayak-baş mesafesi ──
        cap = CONTACT_GAP_CAP  # ayak hiç görülemeyen kareler bu değerde gösterilir
        near, contact = 1.0 + NEAR_MISS_MARGIN_RADII, 1.0 + CONTACT_MARGIN_RADII
        ax2.axhspan(0, contact, color=theme.GOOD, alpha=0.16, linewidth=0)
        ax2.axhspan(contact, near, color=theme.WARN, alpha=0.12, linewidth=0)
        ax2.text(n - 0.5, near + 0.4, "yeşil: temas  ·  sarı: yakın geçiş", fontsize=7.5,
                 color=theme.TEXT_2, ha="right", va="bottom")
        d = np.array([e.get("distance_radii", np.inf) for e in frame_log], dtype=float)
        d[~np.isfinite(d)] = np.nan
        if not np.isnan(d).all():
            ax2.plot(x, np.minimum(d, cap), color=theme.TEXT, linewidth=2)
            ax2.scatter(x, np.minimum(d, cap), s=10, color=theme.TEXT, zorder=3, linewidths=0)
        else:
            ax2.text(0.5, 0.6, "Ayak ve baş birlikte görülemedi", ha="center", va="center",
                     color=theme.TEXT_2, transform=ax2.transAxes, fontsize=9)
        ax2.set_ylim(0, cap)
        ax2.set_title("En yakın ayağın rakip başa uzaklığı", fontsize=10, color=theme.TEXT,
                      loc="left", fontweight="bold")
        ax2.set_ylabel("baş yarıçapı", fontsize=8, color=theme.TEXT_2)
        ax2.set_xlabel("Kare", fontsize=8, color=theme.TEXT_2)
        ax2.set_xlim(-0.5, n - 0.5)

        # Açıklama satırı
        notes = []
        if moment.get("frame") is not None:
            notes.append("kesikli çizgi: temas anı")
        if ranges:
            notes.append("mavi alan: işaretlenen kareler")
        notes.append("grafiğe tıklayın: video o kareye gider")
        self.figure.text(0.01, 0.005, "  ·  ".join(notes), fontsize=7.5, color=theme.MUTED)

        # Videoyla eşzamanlı ilerleyen "şu anki kare" çizgisi
        self._frame_lines = [ax.axvline(0, color=theme.WARN, linewidth=1.6) for ax in (ax1, ax2)]
        self.figure.tight_layout(pad=1.0, rect=(0, 0.03, 1, 1))
        self.canvas.draw()

    def _on_canvas_click(self, event):
        """Grafiğe tıklanınca video o kareye gider."""
        if event.inaxes is None or event.xdata is None or not self._row_for_frame:
            return
        frame = int(round(event.xdata))
        if frame in self._row_for_frame:
            self.frame_selected.emit(frame)

    # ────────────────────────────────────
    # KARE TABLOSU
    # ────────────────────────────────────
    STATUS_TR = {
        "overlap": ("Temas", theme.GOOD),
        "near_miss": ("Yakın geçiş", theme.WARN),
        "no_contact": ("Uzak", None),
        "no_detection": ("Görülemedi", None),
    }

    def _populate_table(self, frame_log: list, rotation: dict, moment_frame=None):
        """Kare kare durum tablosunu doldurur."""
        self._row_for_frame = {}
        if not frame_log:
            self.table.setRowCount(0)
            return

        columns = ["Kare", "Zaman", "Durum", "Ayak-baş\nuzaklığı", "Dönüş\nkırmızı", "Dönüş\nmavi"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(frame_log))

        def fmt_rot(color, i):
            s = rotation.get(color) or []
            v = s[i] if i < len(s) else None
            return "–" if v is None else f"{v:.2f}"

        bold = QFont()
        bold.setBold(True)
        for row, entry in enumerate(frame_log):
            frame = int(entry.get("frame", row))
            self._row_for_frame[frame] = row
            status, status_color = self.STATUS_TR.get(entry.get("contact_type", ""), ("–", None))
            dist = entry.get("distance_radii", float("inf"))
            items = [
                str(frame),
                f"{entry.get('time_sec', 0):.2f} sn",
                status + ("  ● temas anı" if frame == moment_frame else ""),
                f"{dist:.1f}" if np.isfinite(dist) else "–",
                fmt_rot("red", row),
                fmt_rot("blue", row),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if status_color:
                    c = QColor(status_color)
                    c.setAlpha(45)
                    item.setBackground(c)
                if col == 2 and status_color:
                    item.setForeground(QColor(status_color))
                if frame == moment_frame:
                    item.setFont(bold)
                self.table.setItem(row, col, item)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setMinimumHeight(40)
        self.table.verticalHeader().setDefaultSectionSize(26)
        tips = ["Videodaki kare numarası", "Videonun başından itibaren süre",
                "Ayak ile rakip baş arasındaki uzaklığa göre durum",
                "En yakın ayağın rakip baş merkezine uzaklığı (baş yarıçapı cinsinden)",
                "Kırmızı sporcunun baş dönüşü (baş yarıçapı)", "Mavi sporcunun baş dönüşü (baş yarıçapı)"]
        for col, tip in enumerate(tips):
            self.table.horizontalHeaderItem(col).setToolTip(tip)

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
