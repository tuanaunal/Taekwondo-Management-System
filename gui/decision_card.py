"""
decision_card.py — Karar Kartı
===============================
Sonuç panelinin en üstündeki kart. Ziyaretçinin ilk bakışta üç soruya cevap
bulması hedeflenir:

  1. Karar ne?            -> büyük, renkli karar başlığı + görüntü/sensör özeti
  2. Ne kadar emin?       -> modelin temas olasılığı çubuğu + kalibre güven rozeti
  3. Neden?               -> kararı etkileyen ölçümler, temas / temassızlık lehine çubuklar
"""

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QGridLayout

from gui import theme
from src.decision_engine import FEATURE_LABELS_TR, FEATURE_UNITS_TR

CONTACT_LABELS = {"TEMAS_VAR": "Temas var", "TEMAS_YOK": "Temas yok", "BELIRSIZ": "Belirsiz"}
SENSOR_LABELS = {True: "Puan verdi", False: "Puan vermedi", None: "Bilinmiyor"}
UNIT_SHORT = {"baş yarıçapı": "baş yarıçapı", "referans px": "ref. px"}


class ProbabilityBar(QWidget):
    """0-1 arası olasılığı, ortada karar sınırı (0.5) olan yatay bir çubukla gösterir."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = None
        self.color = QColor(theme.ACCENT)
        self.setMinimumHeight(16)
        self.setMaximumHeight(16)

    def set_value(self, value, color: str):
        self.value = value
        self.color = QColor(color)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0, 3, self.width(), 10)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.SURFACE_2))
        p.drawRoundedRect(r, 5, 5)
        if self.value is not None:
            fill = QRectF(r.x(), r.y(), max(10.0, r.width() * float(self.value)), r.height())
            p.setBrush(self.color)
            p.drawRoundedRect(fill, 5, 5)
        # Karar sınırı (0.5)
        p.setPen(QPen(QColor(theme.TEXT), 2))
        x = r.x() + r.width() / 2
        p.drawLine(int(x), 0, int(x), self.height())
        p.end()


class ContributionBar(QWidget):
    """
    Bir ölçümün karara katkısını ortadan iki yöne uzanan çubukla gösterir:
    sağa (yeşil) temas lehine, sola (kırmızı) temassızlık lehine.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.scale = 1.0
        self.setMinimumHeight(14)
        self.setMaximumHeight(14)
        self.setMinimumWidth(90)

    def set_value(self, value: float, scale: float):
        self.value = float(value)
        self.scale = max(float(scale), 1e-6)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = w / 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.SURFACE_2))
        p.drawRoundedRect(QRectF(0, 2, w, h - 4), 4, 4)
        length = min(1.0, abs(self.value) / self.scale) * (mid - 2)
        color = QColor(theme.GOOD if self.value > 0 else theme.BAD)
        p.setBrush(color)
        if self.value > 0:
            p.drawRoundedRect(QRectF(mid, 2, length, h - 4), 4, 4)
        elif self.value < 0:
            p.drawRoundedRect(QRectF(mid - length, 2, length, h - 4), 4, 4)
        p.setPen(QPen(QColor(theme.MUTED), 1))
        p.drawLine(int(mid), 0, int(mid), h)
        p.end()


def _label(text="", role=None, size=None, bold=False, color=None) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setProperty("role", role)
    style = []
    if size:
        style.append(f"font-size: {size}px;")
    if bold:
        style.append("font-weight: 700;")
    if color:
        style.append(f"color: {color};")
    if style:
        lbl.setStyleSheet(" ".join(style))
    return lbl


class DecisionCard(QFrame):
    """Kararı, olasılığı, güveni ve gerekçeyi gösteren kart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        layout.addWidget(_label("KARAR", role="section"))

        # ── Karar başlığı ──
        self.banner = QFrame()
        banner_layout = QVBoxLayout(self.banner)
        banner_layout.setContentsMargins(16, 12, 16, 12)
        banner_layout.setSpacing(2)
        self.title = _label("Analiz bekleniyor", size=24, bold=True, color=theme.TEXT_2)
        self.title.setAlignment(Qt.AlignCenter)
        self.subtitle = _label("Video açıp analizi başlatın", role="muted")
        self.subtitle.setAlignment(Qt.AlignCenter)
        banner_layout.addWidget(self.title)
        banner_layout.addWidget(self.subtitle)
        layout.addWidget(self.banner)

        # ── Olasılık ve güven ──
        self.details = QWidget()
        d = QVBoxLayout(self.details)
        d.setContentsMargins(0, 0, 0, 0)
        d.setSpacing(4)

        prob_head = QHBoxLayout()
        prob_head.addWidget(_label("Modelin temas olasılığı", role="muted"))
        prob_head.addStretch()
        self.prob_value = _label("", bold=True, size=13)
        prob_head.addWidget(self.prob_value)
        d.addLayout(prob_head)
        self.prob_bar = ProbabilityBar()
        d.addWidget(self.prob_bar)
        ends = QHBoxLayout()
        ends.addWidget(_label("temas yok", role="muted"))
        ends.addStretch()
        ends.addWidget(_label("karar sınırı %50", role="muted"))
        ends.addStretch()
        ends.addWidget(_label("temas var", role="muted"))
        d.addLayout(ends)

        conf_row = QHBoxLayout()
        conf_row.setSpacing(8)
        conf_row.addWidget(_label("Güven", role="muted"))
        self.conf_badge = QLabel("")
        conf_row.addWidget(self.conf_badge)
        self.conf_text = _label("", role="muted")
        self.conf_text.setWordWrap(True)
        conf_row.addWidget(self.conf_text, 1)
        d.addSpacing(6)
        d.addLayout(conf_row)

        # ── Gerekçe: ölçümlerin katkısı ──
        d.addSpacing(8)
        self.factors = QWidget()
        f = QVBoxLayout(self.factors)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(4)
        f.addWidget(_label("KARARI ETKİLEYEN ÖLÇÜMLER", role="section"))
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(6)
        self.rows = []
        for i in range(3):
            name = _label("")
            value = _label("", role="muted")
            bar = ContributionBar()
            self.grid.addWidget(name, i, 0)
            self.grid.addWidget(value, i, 1)
            self.grid.addWidget(bar, i, 2)
            self.rows.append((name, value, bar))
        self.grid.setColumnStretch(2, 1)
        f.addLayout(self.grid)
        legend = QHBoxLayout()
        legend.addStretch()
        legend.addWidget(_label("◀ temassızlık lehine", size=10, color=theme.BAD))
        legend.addSpacing(10)
        legend.addWidget(_label("temas lehine ▶", size=10, color=theme.GOOD))
        f.addLayout(legend)
        d.addWidget(self.factors)

        self.note = _label("", role="muted")
        self.note.setWordWrap(True)
        d.addWidget(self.note)

        layout.addWidget(self.details)
        self.clear()

    # ────────────────────────────────────
    def clear(self):
        self._set_banner(theme.NEUTRAL, "Analiz bekleniyor", "Video açıp analizi başlatın", dim=True)
        self.details.setVisible(False)

    def _set_banner(self, color: str, title: str, subtitle: str, dim=False):
        self.banner.setStyleSheet(
            f"QFrame {{ background-color: {theme.rgba(color, 0.08 if dim else 0.14)};"
            f" border: 1px solid {theme.rgba(color, 0.25 if dim else 0.55)}; border-radius: 10px; }}"
            " QLabel { border: none; background: transparent; }")
        self.title.setText(title)
        self.title.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {theme.TEXT_2 if dim else color};")
        self.subtitle.setText(subtitle)

    def display(self, decision: dict):
        code = decision.get("decision", "BELIRSIZ")
        color = theme.DECISION_COLORS.get(code, theme.NEUTRAL)
        contact = decision.get("contact_decision", code)
        sensor = decision.get("sensor_scored")
        subtitle = (f"Görüntü: {CONTACT_LABELS.get(contact, contact)}   •   "
                    f"Sensör: {SENSOR_LABELS.get(sensor, 'Bilinmiyor')}")
        self._set_banner(color, f"{decision.get('indicator', '')}  {decision.get('label_tr', '')}", subtitle)

        prob = decision.get("contact_probability")
        features = decision.get("features") or {}
        contributions = decision.get("contributions") or {}
        self.details.setVisible(True)

        if prob is None or not contributions:
            # Görüntü yetersiz: model çalıştırılmadı
            self.prob_value.setText("–")
            self.prob_value.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {theme.TEXT_2};")
            self.prob_bar.set_value(None, theme.NEUTRAL)
            self.factors.setVisible(False)
            self._set_confidence(None, None)
            self.note.setText(decision.get("reasoning", ""))
            return

        self.factors.setVisible(True)
        prob = float(prob)
        prob_color = theme.GOOD if prob >= 0.5 else theme.BAD
        self.prob_value.setText(f"%{prob * 100:.0f}")
        self.prob_value.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {prob_color};")
        self.prob_bar.set_value(prob, prob_color)

        ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        scale = max(abs(v) for _, v in ordered) if ordered else 1.0
        scale = max(scale, 1.0)
        for (name_lbl, value_lbl, bar), (key, c) in zip(self.rows, ordered):
            label = FEATURE_LABELS_TR.get(key, key)
            unit = UNIT_SHORT.get(FEATURE_UNITS_TR.get(key, ""), "")
            v = features.get(key)
            name_lbl.setText(label[:1].upper() + label[1:])
            value_lbl.setText(f"{v:.0f} {unit}" if v is not None and abs(v) >= 100
                              else (f"{v:.2f} {unit}" if v is not None else ""))
            bar.set_value(c, scale)
            bar.setToolTip(f"Katkı: {c:+.2f} ({'temas' if c > 0 else 'temassızlık'} lehine)")

        self._set_confidence(decision.get("confidence"), decision.get("confidence_level"))
        self.note.setText("Nihai karar hakemindir; sistem ikinci görüş sunar.")

    def _set_confidence(self, confidence, level):
        if confidence is None or not level or level == "-":
            self.conf_badge.setText(" – ")
            self.conf_badge.setStyleSheet(
                f"background-color: {theme.SURFACE_2}; color: {theme.TEXT_2}; border-radius: 8px;"
                " padding: 2px 10px; font-weight: 700;")
            self.conf_text.setText("Görüntü karar vermeye yetmedi.")
            return
        c = theme.CONFIDENCE_COLORS.get(level, theme.NEUTRAL)
        self.conf_badge.setText(f"{level.upper()}  %{confidence * 100:.0f}")
        self.conf_badge.setStyleSheet(
            f"background-color: {theme.rgba(c, 0.18)}; color: {c}; border: 1px solid {theme.rgba(c, 0.6)};"
            " border-radius: 8px; padding: 2px 10px; font-weight: 700;")
        self.conf_text.setText(f"Benzer kararlar, hiç görülmemiş kliplerde %{confidence * 100:.0f} oranında doğru çıktı.")
