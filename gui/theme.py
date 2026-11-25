"""
theme.py — Arayüzün Ortak Renk ve Stil Tanımları
=================================================
Tüm renkler burada tek bir yerde tanımlanır; pencereler ve paneller kendi
renklerini yazmak yerine bu değerleri ve APP_STYLESHEET'i kullanır.

Stil sınıfları (widget.setProperty ile verilir):
  QPushButton[variant="primary"]   -> ana eylem (Analizi Başlat)
  QPushButton[variant="secondary"] -> ikincil eylem (Video Aç, Rapor Kaydet)
  QPushButton[variant="ghost"]     -> küçük araç düğmeleri (oynatma, hız, yakınlaştırma)
  QFrame#card                      -> kart görünümlü bölüm
  QLabel[role="section"]           -> bölüm başlığı (küçük, büyük harf)
  QLabel[role="muted"]             -> ikincil açıklama metni
"""

# ── Yüzeyler ve metin ──
BG = "#0b0f17"          # pencere zemini
SURFACE = "#121826"     # kartlar
SURFACE_2 = "#1a2233"   # kart içindeki kutular, düğmeler
BORDER = "#263043"
TEXT = "#e6e9ef"
TEXT_2 = "#9aa4b2"
MUTED = "#6b7585"

# ── Vurgu ──
ACCENT = "#3b82f6"       # ana eylem
ACCENT_HOVER = "#5b9bff"
BRAND_RED = "#e5484d"    # taekwondo kırmızısı (başlık amblemi, kırmızı sporcu)
BRAND_BLUE = "#3b82f6"   # mavi sporcu

# ── Durum renkleri (karar kartı) ──
GOOD = "#22c55e"
WARN = "#f59e0b"
BAD = "#ef4444"
NEUTRAL = "#94a3b8"
CAUTION = "#f97316"      # düşük güven

# Karar kodu -> durum rengi
DECISION_COLORS = {
    "PUAN_GECERLI": GOOD, "TEMAS_VAR": GOOD, "DOGRU_KARAR": GOOD,
    "GHOST_HIT": BAD, "TEMAS_YOK": BAD,
    "KACIRILAN_VURUS": WARN,
    "BELIRSIZ": NEUTRAL,
}

# Güven seviyesi -> renk
CONFIDENCE_COLORS = {"yüksek": GOOD, "orta": WARN, "düşük": CAUTION}


def rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' rengini Qt stil sayfası için 'rgba(r, g, b, a)' biçimine çevirir."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {int(alpha * 255)})"


APP_STYLESHEET = f"""
QMainWindow, QWidget#central {{
    background-color: {BG};
}}
QWidget {{
    color: {TEXT};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 12px;
}}
QToolTip {{
    background-color: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px;
}}

/* ── Kartlar ve başlıklar ── */
QFrame#card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#card QLabel {{
    background: transparent;
}}
QLabel[role="section"] {{
    color: {TEXT_2};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel[role="muted"] {{
    color: {TEXT_2};
    font-size: 11px;
}}
QLabel[role="step"] {{
    background-color: {rgba(ACCENT, 0.18)};
    color: {ACCENT_HOVER};
    border-radius: 11px;
    font-size: 11px;
    font-weight: 700;
    min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px;
}}

/* ── Düğmeler ── */
QPushButton {{
    background-color: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {BORDER};
}}
QPushButton:disabled {{
    color: {MUTED};
    border-color: {SURFACE_2};
}}
QPushButton[variant="primary"] {{
    background-color: {ACCENT};
    border: none;
    color: white;
    font-size: 14px;
    font-weight: 700;
    padding: 12px 16px;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton[variant="primary"]:disabled {{
    background-color: {SURFACE_2};
    color: {MUTED};
}}
QPushButton[variant="secondary"] {{
    font-size: 13px;
    padding: 10px 14px;
}}
QPushButton[variant="ghost"] {{
    background-color: transparent;
    border: 1px solid {BORDER};
    padding: 6px 10px;
    font-size: 12px;
}}
QPushButton[variant="ghost"]:hover {{
    background-color: {SURFACE_2};
}}
QPushButton[variant="ghost"]:checked {{
    background-color: {rgba(ACCENT, 0.22)};
    border-color: {ACCENT};
    color: white;
}}

/* ── Giriş alanları ── */
QComboBox {{
    background-color: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 10px;
}}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QProgressBar {{
    background-color: {SURFACE_2};
    border: none;
    border-radius: 6px;
    height: 12px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 6px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {SURFACE_2};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: white;
    border: 3px solid {ACCENT};
    width: 12px;
    margin: -6px 0;
    border-radius: 9px;
}}

/* ── Eski bölüm kutuları (grafikler / tablo) ── */
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    margin-top: 14px;
    padding: 18px 8px 8px 8px;
    color: {TEXT_2};
    font-size: 11px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {TEXT_2};
}}

/* ── Sekmeler ── */
QTabWidget::pane {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_2};
    padding: 8px 16px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-bottom-color: {SURFACE};
}}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

/* ── Tablo ── */
QTableWidget {{
    background-color: {SURFACE};
    alternate-background-color: {SURFACE_2};
    color: {TEXT};
    gridline-color: {BORDER};
    border: none;
    font-size: 11px;
}}
QTableWidget::item:selected {{
    background-color: {rgba(ACCENT, 0.45)};
    color: white;
}}
QHeaderView::section {{
    background-color: {SURFACE_2};
    color: {TEXT_2};
    padding: 6px 4px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 700;
    font-size: 11px;
}}
QTableCornerButton::section {{
    background-color: {SURFACE_2};
    border: none;
}}

/* ── Kaydırma çubukları, ayırıcı, durum çubuğu ── */
QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    width: 10px; height: 10px;
    border: none;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
QSplitter::handle {{
    background-color: {BG};
}}
QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT_2};
    border-top: 1px solid {BORDER};
    font-size: 11px;
}}
QStatusBar QLabel {{ color: {TEXT_2}; }}
QMessageBox {{ background-color: {SURFACE}; }}
"""
