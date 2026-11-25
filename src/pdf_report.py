"""
pdf_report.py — Analiz Raporu (PDF)
====================================
Bir analizin sonucunu, temas anının görüntüsü, grafikler, kararı belirleyen
ölçümler ve sistemin sınırlarıyla birlikte tek sayfalık bir PDF rapora dönüştürür.

Türkçe karakterler için Windows'taki Arial yazı tipi kullanılır; bulunamazsa
karakterler en yakın Latin karşılıklarına çevrilir (ı -> i, ş -> s ...).
"""

import os
import shutil
import tempfile
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from gui.overlay import draw_frame_features
from src.config import CONTACT_MARGIN_RADII, NEAR_MISS_MARGIN_RADII
from src.decision_engine import CONTACT_GAP_CAP, FEATURE_LABELS_TR, FEATURE_UNITS_TR, load_model

FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
TR_FALLBACK = str.maketrans("ıİşŞğĞüÜöÖçÇâÂ–—’“”", "iIsSgGuUoOcCaA--'\"\"")
SENSOR_TR = {True: "Puan verdi", False: "Puan vermedi", None: "Bilinmiyor"}


class _Report(FPDF):
    def __init__(self, unicode_font: bool):
        super().__init__()
        self.unicode_font = unicode_font
        self.font_name = "TR" if unicode_font else "Arial"

    def txt(self, s) -> str:
        s = str(s)
        return s if self.unicode_font else s.translate(TR_FALLBACK).encode("latin-1", "ignore").decode("latin-1")

    def header(self):
        self.set_fill_color(22, 33, 62)
        self.rect(0, 0, 210, 22, "F")
        self.set_xy(10, 5)
        self.set_font(self.font_name, "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, self.txt("Taekwondo Hakem Karar Destek Sistemi — Analiz Raporu"), 0, 1, "L")
        self.set_x(10)
        self.set_font(self.font_name, "", 9)
        self.set_text_color(200, 200, 200)
        self.cell(0, 5, self.txt("Elektronik puanlama hatalarının (Ghost Hit / kaçırılan vuruş) "
                                 "bilgisayarlı görü ile incelenmesi"), 0, 1, "L")
        self.set_y(26)

    def footer(self):
        self.set_y(-12)
        self.set_font(self.font_name, "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 6, self.txt(f"Sayfa {self.page_no()}  |  Oluşturulma: "
                                 f"{datetime.now().strftime('%d.%m.%Y %H:%M')}  |  "
                                 "Bu rapor bir karar destek aracının çıktısıdır; hakem kararının yerine geçmez."),
                  0, 0, "C")

    def section(self, title: str):
        self.ln(2)
        self.set_font(self.font_name, "B", 11)
        self.set_text_color(22, 33, 62)
        self.cell(0, 7, self.txt(title), 0, 1)
        self.set_draw_color(206, 212, 218)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(1.5)

    def row(self, key: str, value: str, key_w: float = 48):
        self.set_font(self.font_name, "B", 9)
        self.set_text_color(73, 80, 87)
        self.cell(key_w, 5.5, self.txt(key))
        self.set_font(self.font_name, "", 9)
        self.cell(0, 5.5, self.txt(value), 0, 1)


def _make_pdf() -> _Report:
    for regular, bold in FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            # fpdf 1.7.2 yazı tipi önbelleğini yazı tipinin klasörüne yazar; sistem klasörü yazılamayabilir
            font_dir = os.path.join(tempfile.gettempdir(), "tkd_rapor_fontlari")
            os.makedirs(font_dir, exist_ok=True)
            paths = []
            for src in (regular, bold):
                dst = os.path.join(font_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                paths.append(dst)
            pdf = _Report(unicode_font=True)
            pdf.add_font("TR", "", paths[0], uni=True)
            pdf.add_font("TR", "B", paths[1], uni=True)
            return pdf
    return _Report(unicode_font=False)


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _contact_frame_image(video_path: str, result: dict, path: str):
    """Temas anı karesini baş/ayak işaretleriyle kaydeder; kare yoksa False."""
    frame_idx = result.get("contact_moment", {}).get("frame")
    feats = result.get("frame_features", [])
    if frame_idx is None or not video_path or not os.path.exists(video_path) or frame_idx >= len(feats):
        return False
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    img = draw_frame_features(frame, feats[frame_idx], result.get("video_info", {}).get("scale_factor", 1.0))
    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def _graphs_image(result: dict, path: str):
    """
    Arayüzdeki grafiklerin aynısını (açık renkli, baskıya uygun) kaydeder:
      - kare kare baş dönüşü (kırmızı / mavi sporcu) — kararın en etkili ölçümü
      - en yakın ayağın rakip baş merkezine uzaklığı (baş yarıçapı), temas / yakın geçiş bölgeleriyle
    Temas anı kesikli çizgi, hakemin işaretlediği kareler gölgeli alan olarak gösterilir.
    """
    frame_log = result.get("frame_log", [])
    head = result.get("head_response", {}) or {}
    moment = (result.get("contact_moment", {}) or {}).get("frame")
    ranges = result.get("contact_ranges") or []
    n = len(frame_log)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.3), dpi=110)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=7)
        for a, b in ranges:
            ax.axvspan(a - 0.5, b + 0.5, color="#2a78d6", alpha=0.12, linewidth=0)
        if moment is not None:
            ax.axvline(moment, color="#52514e", linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlabel("Kare", fontsize=8)
        ax.set_ylabel("baş yarıçapı", fontsize=8)
        if n:
            ax.set_xlim(-0.5, n - 0.5)

    # Baş dönüşü
    drawn = False
    for color, line_color, name in (("red", "#e5484d", "Kırmızı"), ("blue", "#2a78d6", "Mavi")):
        series = (head.get(color, {}) or {}).get("rotation_series") or []
        y = np.array([np.nan if v is None else v for v in series[:n]], dtype=float)
        if y.size and not np.isnan(y).all():
            axes[0].plot(np.arange(y.size), y, color=line_color, linewidth=1.6, label=name)
            drawn = True
    axes[0].set_title("Baş dönüşü", fontsize=10, loc="left")
    if drawn:
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0)

    # Ayak-baş uzaklığı
    cap = CONTACT_GAP_CAP
    contact, near = 1.0 + CONTACT_MARGIN_RADII, 1.0 + NEAR_MISS_MARGIN_RADII
    axes[1].axhspan(0, contact, color="#22c55e", alpha=0.18, linewidth=0, label="temas")
    axes[1].axhspan(contact, near, color="#f59e0b", alpha=0.18, linewidth=0, label="yakın geçiş")
    d = np.array([e.get("distance_radii", np.inf) for e in frame_log], dtype=float)
    d[~np.isfinite(d)] = np.nan
    if d.size and not np.isnan(d).all():
        axes[1].plot(np.arange(d.size), np.minimum(d, cap), color="#0b0b0b", linewidth=1.4, marker="o",
                     markersize=2)
    axes[1].set_ylim(0, cap)
    axes[1].set_title("En yakın ayağın rakip başa uzaklığı", fontsize=10, loc="left")
    axes[1].legend(fontsize=7, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def create_pdf_report(result: dict, video_path: str, out_path: str) -> str:
    """
    Analiz sonucundan PDF rapor oluşturur.

    Parameters
    ----------
    result : main.analyze_video() çıktısı
    video_path : analiz edilen video (temas anı görüntüsü için)
    out_path : kaydedilecek PDF yolu
    """
    decision = result.get("decision_result", {})
    moment = result.get("contact_moment", {})
    info = result.get("video_info", {})
    pdf = _make_pdf()
    pdf.set_auto_page_break(True, margin=14)
    pdf.add_page()

    # ── Video bilgisi ──
    pdf.section("Video")
    pdf.row("Dosya:", os.path.basename(video_path or "-"))
    pdf.row("Çözünürlük / FPS:", f"{info.get('width', '-')}x{info.get('height', '-')}  /  "
                                 f"{info.get('fps', 0):.0f} FPS  ({info.get('frame_count', '-')} kare)")

    # ── Karar ──
    pdf.section("Karar")
    r, g, b = _hex_to_rgb(decision.get("color", "#95A5A6"))
    pdf.set_fill_color(int(r + (255 - r) * 0.85), int(g + (255 - g) * 0.85), int(b + (255 - b) * 0.85))
    pdf.set_text_color(int(r * 0.7), int(g * 0.7), int(b * 0.7))
    pdf.set_font(pdf.font_name, "B", 13)
    level = decision.get("confidence_level", "-")
    pdf.cell(0, 11, pdf.txt(f"  {decision.get('label_tr', '-')}   —   Güven: "
                            f"%{decision.get('confidence', 0) * 100:.0f} ({level})"), 0, 1, "L", fill=True)
    pdf.ln(1)
    pdf.row("Sensör durumu:", SENSOR_TR.get(decision.get("sensor_scored")))
    pdf.row("Görüntü kararı:", {"TEMAS_VAR": "Temas var", "TEMAS_YOK": "Temas yok",
                                "BELIRSIZ": "Belirsiz"}.get(decision.get("contact_decision"), "-"))
    if moment.get("frame") is not None:
        pdf.row("Temas anı:", f"Kare {moment['frame']} ({moment.get('time_sec', 0):.2f} sn) — "
                              f"{moment.get('method_tr', '')}")
    pdf.set_font(pdf.font_name, "", 9)
    pdf.set_text_color(73, 80, 87)
    pdf.multi_cell(0, 5, pdf.txt("Gerekçe: " + decision.get("reasoning", "")))

    # ── Temas anı görüntüsü ve grafikler ──
    tmp = tempfile.mkdtemp(prefix="tkd_rapor_")
    try:
        img_path = os.path.join(tmp, "temas.jpg")
        if _contact_frame_image(video_path, result, img_path):
            pdf.section("Temas anının görüntüsü (daire: baş bölgesi, nokta: ayaklar / burun)")
            pdf.image(img_path, x=57, w=96)
        graph_path = os.path.join(tmp, "grafik.png")
        _graphs_image(result, graph_path)
        pdf.section("Grafikler (kesikli çizgi: temas anı, gölgeli alan: işaretlenen kareler)")
        pdf.image(graph_path, x=10, w=190)

        # ── Ölçümler ──
        pdf.section("Kararı belirleyen ölçümler")
        feats = decision.get("features", {})
        contrib = decision.get("contributions", {})
        for name, value in feats.items():
            c = contrib.get(name)
            effect = "" if c is None else (f"  (karara etkisi {c:+.2f}: "
                                           f"{'temas lehine' if c > 0 else 'temassızlık lehine'})")
            pdf.row(FEATURE_LABELS_TR.get(name, name) + ":", f"{value:.2f} {FEATURE_UNITS_TR.get(name, '')}{effect}")

        # ── Sınırlar ──
        pdf.section("Sınırlar")
        model = load_model() or {}
        v = model.get("dogrulama", {})
        acc = v.get("orijinal_20_dogruluk")
        acc_txt = f"%{acc * 100:.0f}" if acc is not None else "bilinmiyor"
        pdf.set_font(pdf.font_name, "", 8.5)
        pdf.set_text_color(73, 80, 87)
        pdf.multi_cell(0, 4.6, pdf.txt(
            f"Karar modeli {model.get('egitim_video_sayisi', '-')} video ile eğitilmiştir. Her klip bir kez dışarıda "
            f"tutularak ölçülen dürüst doğruluğu {acc_txt}'dir; gösterilen güven, benzer kararların bu ölçümdeki "
            "gerçek doğruluk oranıdır. Tek kameralı iki boyutlu görüntüde temas ile temassız geçiş her zaman ayırt "
            "edilemez ve temas anında iskelet noktaları güvenilirliğini yitirebilir. Bu rapor hakeme ikinci bir "
            "görüş sunar; nihai karar hakeme aittir."))
        pdf.output(out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path
