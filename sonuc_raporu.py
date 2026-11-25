"""
sonuc_raporu.py — Sonuç Raporu Üretici
=======================================
egit.py'nin çıkardığı ölçümlerden (output/reports/ozellikler.csv) sistemin DÜRÜST
sonuçlarını hesaplar ve sunumda kullanılacak tek bir rapor üretir:

  - SONUC_RAPORU.md        : doğruluk, karmaşıklık matrisi, klip klip sonuçlar, hatalar
  - sonuc/*.png            : karmaşıklık matrisi, klip olasılıkları, doğruluk geçmişi

Tüm tahminler "birini dışarıda bırakarak" elde edilir: her klip tüm çoğaltılmış
kopyalarıyla birlikte bir kez dışarıda tutulur, model kalan kliplerle eğitilir.
Yani buradaki hiçbir sonuç, modelin eğitimde gördüğü bir klipten gelmez.

Kullanım:
    python egit.py --onbellek     (veya ölçümler yoksa: python egit.py)
    python sonuc_raporu.py
"""

import csv
import io
import os
import sys
from datetime import datetime
from math import comb, sqrt

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egit import (AUGMENT_TAGS, CACHE_PATH, DECISION_THRESHOLD, calibration_table,
                  leave_one_clip_out, load_cache)
from src.config import PROJECT_ROOT, REPORTS_DIR
from src.decision_engine import FEATURES, FEATURE_LABELS_TR, MIN_HEAD_COVERAGE

OUT_MD = os.path.join(PROJECT_ROOT, "SONUC_RAPORU.md")
OUT_DIR = os.path.join(PROJECT_ROOT, "sonuc")
HISTORY_PATH = os.path.join(REPORTS_DIR, "degerlendirme_gecmisi.csv")

# Renkler (doğrulanmış kategorik palet; sınıf kimliği her yerde aynı renkte)
C_GHOST, C_REAL = "#eb6834", "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

YES, NO, UNC = "Temas var", "Temas yok", "Belirsiz"


def variant(name: str) -> str:
    base = os.path.splitext(name)[0]
    for t in AUGMENT_TAGS:
        if base.endswith("_" + t):
            return t
    return "orijinal"


VARIANT_TR = {"orijinal": "Orijinal", "flipped": "Aynalama", "bright": "Parlak",
              "dark": "Karanlık", "noisy": "Gürültülü", "blur": "Bulanık"}


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def p_at_least(k: int, n: int) -> float:
    """Yazı-tura atan bir sistemin en az k doğru yapma olasılığı (tek yönlü binom testi)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def decide(prob: float, coverage: float) -> str:
    if coverage < MIN_HEAD_COVERAGE:
        return UNC
    return YES if prob >= DECISION_THRESHOLD else NO


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_confusion(cm: np.ndarray, cols: list, path: str, title: str):
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    vmax = max(cm.max(), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = SEQ[int(round(v / vmax * (len(SEQ) - 1)))]
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, color=color, lw=0))
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=16,
                    color="#ffffff" if v / vmax > 0.55 else INK, fontweight="bold")
    ax.set_xlim(0, cm.shape[1])
    ax.set_ylim(cm.shape[0], 0)
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5, cols)
    ax.set_yticks([0.5, 1.5], ["Ghost Hit\n(gerçekte temas yok)", "Real Hit\n(gerçekte temas var)"])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, colors=INK, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, pad=28, loc="left")
    fig.text(0.5, 0.02, "Sütun: sistemin kararı   ·   Satır: gerçek durum", ha="center",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_clip_probs(clips: list, path: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for idx, c in enumerate(clips):
        col = C_REAL if c["etiket"] == 1 else C_GHOST
        ax.plot([idx, idx], [min(c["kopya_olasiliklari"]), max(c["kopya_olasiliklari"])],
                color=col, lw=2, alpha=0.35, solid_capstyle="round")
        ax.scatter(idx, c["olasilik"], s=46, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        if not c["dogru"]:
            ax.scatter(idx, c["olasilik"], s=150, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.axhline(DECISION_THRESHOLD, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(len(clips) - 0.4, DECISION_THRESHOLD + 0.015, "karar sınırı", fontsize=8,
            color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(len(clips)), [c["klip"] for c in clips], rotation=60, fontsize=8)
    ax.set_ylabel("Model: \"temas var\" olasılığı", fontsize=9, color=INK2)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=C_GHOST, label="Ghost Hit (doğrusu: altta)"),
               Line2D([], [], marker="o", ls="", color=C_REAL, label="Real Hit (doğrusu: üstte)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK,
                      markersize=10, label="Yanlış karar")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left", ncol=3,
              bbox_to_anchor=(0, 1.13))
    fig.text(0.01, 0.01, "Nokta: orijinal klip · Soluk çizgi: aynı klibin 6 kopyasındaki olasılık aralığı",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_history(path: str, honest_acc: float):
    rows = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r["kapsam"] == "orijinal"]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    x = np.arange(len(rows))
    y = [float(r["dogruluk"]) * 100 for r in rows]
    ax.plot(x, y, color=C_REAL, lw=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    ax.scatter([len(rows)], [honest_acc * 100], s=60, color=INK, zorder=3)
    ax.annotate(f"Dürüst ölçüm\n%{honest_acc * 100:.0f}", (len(rows), honest_acc * 100),
                xytext=(-10, 12), textcoords="offset points", fontsize=8, color=INK, ha="right")
    ax.axhline(50, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0, 51.5, "şans düzeyi (%50)", fontsize=8, color=INK2)
    labels = [r["not"].split(":")[0].replace("adim", "Adım").replace("baslangic (Adım 3)", "Başl.")
              .replace("baslangic (adim 3)", "Başl.") for r in rows] + ["Dürüst"]
    ax.set_xticks(list(x) + [len(rows)], labels, rotation=60, fontsize=7.5)
    ax.set_ylabel("20 orijinal klipte doğruluk (%)", fontsize=9, color=INK2)
    ax.set_ylim(30, 80)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Mavi çizgi: aynı veride ayarlanmış (iyimser) ölçümler · Siyah nokta: görülmemiş klipte",
                 fontsize=8.5, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    if not os.path.exists(CACHE_PATH):
        print("Ölçümler bulunamadı. Önce 'python egit.py' çalıştırın.")
        sys.exit(1)
    rows = load_cache()
    for r in rows:
        r["varyant"] = variant(r["video"])
    probs = leave_one_clip_out(rows, FEATURES)
    for r, p in zip(rows, probs):
        r["olasilik"] = float(p)
        r["karar"] = decide(p, r["kapsama"])
        r["gercek"] = YES if r["etiket"] == 1 else NO
        r["dogru"] = r["karar"] == r["gercek"]

    orig = [r for r in rows if r["orijinal"] == 1]
    n_o, k_o = len(orig), sum(r["dogru"] for r in orig)
    n_a, k_a = len(rows), sum(r["dogru"] for r in rows)
    lo, hi = wilson(k_o, n_o)
    p_chance = p_at_least(k_o, n_o)

    cols = [YES, NO, UNC]
    used_cols = [c for c in cols if c != UNC or any(r["karar"] == UNC for r in rows)]

    def cmatrix(rs):
        m = np.zeros((2, len(used_cols)), dtype=int)
        for r in rs:
            m[r["etiket"] == 1 and 1 or 0, used_cols.index(r["karar"])] += 1
        return m
    cm_o, cm_a = cmatrix(orig), cmatrix(rows)

    # Klip bazında tablo
    clips = []
    for r in sorted(orig, key=lambda r: (r["etiket"], r["klip"])):
        copies = [x for x in rows if x["klip"] == r["klip"]]
        clips.append({**r, "kopya_olasiliklari": [x["olasilik"] for x in copies],
                      "kopya_dogru": sum(x["dogru"] for x in copies), "kopya_sayisi": len(copies)})

    # Varyantlara göre dayanıklılık
    by_var = []
    for v in ["orijinal"] + AUGMENT_TAGS:
        rs = [r for r in rows if r["varyant"] == v]
        if rs:
            by_var.append((VARIANT_TR[v], sum(r["dogru"] for r in rs), len(rs)))

    # Ölçümlerin ayırt ediciliği (orijinal 20)
    feat_stats = []
    for k in FEATURES:
        pos = [r[k] for r in orig if r["etiket"] == 1]
        neg = [r[k] for r in orig if r["etiket"] == 0]
        a = auc(pos, neg)
        feat_stats.append((FEATURE_LABELS_TR[k], np.median(neg), np.median(pos), a))

    calib = calibration_table(rows, probs)

    # Grafikler
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_confusion(cm_o, used_cols, os.path.join(OUT_DIR, "karmasiklik_matrisi_20.png"),
                   f"20 orijinal klip — {k_o}/{n_o} doğru (görülmemiş klip testi)")
    plot_confusion(cm_a, used_cols, os.path.join(OUT_DIR, "karmasiklik_matrisi_120.png"),
                   f"120 video (çoğaltılmışlar dahil) — {k_a}/{n_a} doğru")
    plot_clip_probs(clips, os.path.join(OUT_DIR, "klip_olasiliklari.png"))
    has_hist = plot_history(os.path.join(OUT_DIR, "dogruluk_gecmisi.png"), k_o / n_o)

    gh_caught = cm_o[0, used_cols.index(NO)]
    rh_cancel = cm_o[1, used_cols.index(NO)]
    rh_ok = cm_o[1, used_cols.index(YES)]
    gh_missed = cm_o[0, used_cols.index(YES)]

    wrong = [c for c in clips if not c["dogru"]]
    L = []
    w = L.append
    w("# Sonuç Raporu\n")
    w(f"*Üretildi: {datetime.now().strftime('%Y-%m-%d %H:%M')} · `python sonuc_raporu.py` · "
      f"ölçümler: `{os.path.relpath(CACHE_PATH, PROJECT_ROOT)}`*\n")
    w("Bu rapor, sistemin **hiç görmediği kliplerdeki** başarısını özetler. Her klip, tüm çoğaltılmış "
      "kopyalarıyla birlikte bir kez dışarıda tutulmuş; karar modeli kalan 19 klip ile eğitilip "
      "dışarıdaki klipte test edilmiştir (birini dışarıda bırakarak doğrulama). Eşikler test verisine "
      "göre ayarlanmamıştır.\n")
    w("---\n")
    w("## 1. Özet\n")
    w("| | Sonuç |\n|---|---|")
    w(f"| **Doğruluk (20 orijinal klip)** | **{k_o}/{n_o} (%{k_o / n_o * 100:.0f})** |")
    w(f"| %95 güven aralığı | %{lo * 100:.0f} – %{hi * 100:.0f} |")
    w(f"| Şans düzeyinde bir sistemin bu kadar veya daha iyi sonuç alma olasılığı | %{p_chance * 100:.0f} |")
    w(f"| Doğruluk (120 video, çoğaltılmışlar dahil) | {k_a}/{n_a} (%{k_a / n_a * 100:.0f}) |")
    w(f"| Ghost Hit yakalama (10 klipten) | {gh_caught}/{sum(cm_o[0])} |")
    w(f"| Real Hit'i doğru onaylama (10 klipten) | {rh_ok}/{sum(cm_o[1])} |")
    w("")
    w(f"**Yorum:** {k_o}/{n_o} sonucu şans düzeyinden istatistiksel olarak ayırt edilemiyor "
      f"(yazı-tura atan bir sistem de %{p_chance * 100:.0f} olasılıkla bu sonucu alır). Mevcut ölçümler "
      "20 klipte Real Hit ile Ghost Hit'i güvenilir biçimde ayıramıyor. Sistem bu yüzden her kararı "
      "kalibre edilmiş, çoğunlukla **düşük** bir güvenle verir ve nihai kararı hakeme bırakır.\n")
    w("---\n")
    w("## 2. Karmaşıklık matrisi\n")
    w("Veri setindeki tüm klipler sensörün puan verdiği anlardır. Bu yüzden \"temas yok\" kararı "
      "**Ghost Hit (puan hatalı)**, \"temas var\" kararı **puan geçerli** anlamına gelir.\n")
    hdr = "| Gerçek durum ↓ / Sistem kararı → | " + " | ".join(used_cols) + " |"
    w(hdr)
    w("|---|" + "---|" * len(used_cols))
    w("| **Ghost Hit** (temas yok) | " + " | ".join(str(v) for v in cm_o[0]) + " |")
    w("| **Real Hit** (temas var) | " + " | ".join(str(v) for v in cm_o[1]) + " |")
    w("")
    w("![Karmaşıklık matrisi — 20 orijinal klip](sonuc/karmasiklik_matrisi_20.png)\n")
    w(f"- **Yakalanan Ghost Hit:** {gh_caught} — sistem hatalı puanı doğru tespit etti.")
    w(f"- **Kaçırılan Ghost Hit:** {gh_missed} — hatalı puan geçerli sayıldı.")
    w(f"- **Haksız iptal önerisi:** {rh_cancel} — gerçek bir vuruş için \"puan hatalı\" denildi. "
      "Sahada en maliyetli hata budur; bu yüzden sistemin kararı tek başına puan iptali için kullanılmamalıdır.")
    w(f"- **Doğru onay:** {rh_ok} — gerçek vuruşun puanı onaylandı.\n")
    w("120 videonun tamamında (her klibin 5 çoğaltılmış kopyası dahil):\n")
    w(hdr)
    w("|---|" + "---|" * len(used_cols))
    w("| **Ghost Hit** | " + " | ".join(str(v) for v in cm_a[0]) + " |")
    w("| **Real Hit** | " + " | ".join(str(v) for v in cm_a[1]) + " |")
    w("")
    w("---\n")
    w("## 3. Klip klip sonuçlar\n")
    w("![Klip olasılıkları](sonuc/klip_olasiliklari.png)\n")
    w("| Klip | Gerçek | Karar | Olasılık | Doğru mu | 6 kopyadan doğru | Baş görünürlüğü | "
      "Baş dönüşü | Burun savrulması | Ayak-baş mesafesi (baş yarıçapı) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in clips:
        w(f"| {c['klip']} | {c['gercek']} | {c['karar']} | %{c['olasilik'] * 100:.1f} |"
          f"{'✅' if c['dogru'] else '❌'} | {c['kopya_dogru']}/{c['kopya_sayisi']} | "
          f"%{c['kapsama'] * 100:.0f} | {c['rotation']:.2f} | {c['absolute_swing']:.0f} | "
          f"{c['contact_gap_radii']:.1f} |")
    w("")
    stable = sum(1 for c in clips if c["kopya_dogru"] in (0, c["kopya_sayisi"]))
    accs = [k / n for _, k, n in by_var]
    w(f"**Kararların tutarlılığı:** {stable}/{len(clips)} klipte sistem 6 kopyanın hepsinde aynı "
      f"doğrulukta karar verdi (hepsi doğru veya hepsi yanlış); diğer {len(clips) - stable} klipte "
      "karar kopyadan kopyaya değişiyor (grafikteki uzun soluk çizgiler). Buna karşın her "
      f"görüntü koşulundaki toplam doğruluk benzer (%{min(accs) * 100:.0f}–%{max(accs) * 100:.0f}, "
      "bkz. 4. bölüm): görüntü koşulları başarıyı belirli bir yöne kaydırmıyor; hatalar görüntü "
      "kalitesinden çok ölçümlerin kendisinden geliyor.\n")
    w("---\n")
    w("## 4. Görüntü koşullarına dayanıklılık\n")
    w("| Koşul | Doğru | Doğruluk |\n|---|---|---|")
    for name, k, n in by_var:
        w(f"| {name} | {k}/{n} | %{k / n * 100:.0f} |")
    w("")
    w("---\n")
    w("## 5. Ölçümlerin ayırt ediciliği\n")
    w("AUC: rastgele seçilen bir Real Hit klibinin, rastgele bir Ghost Hit klibinden daha yüksek değer "
      "alma olasılığı (%50 = şans, %100 = kusursuz ayrım). Ayak-baş mesafesinde küçük değer temas lehinedir, "
      "bu yüzden %50'nin altı beklenen yöndür.\n")
    w("| Ölçüm | Ghost Hit ortanca | Real Hit ortanca | AUC (20 klip) |\n|---|---|---|---|")
    for name, mg, mr, a in feat_stats:
        w(f"| {name} | {mg:.2f} | {mr:.2f} | %{a * 100:.0f} |")
    w("")
    w("Baş dönüşü ve burun savrulması 20 klipte bir eğilim gösteriyor; ancak bu eğilim, klip dışarıda "
      "tutulduğunda doğru tahmine dönüşecek kadar güçlü değil (bkz. Özet).\n")
    w("---\n")
    w("## 6. Güven kalibrasyonu\n")
    w("Arayüzde gösterilen güven, modelin kendi olasılığı değil, benzer olasılıktaki kararların bu "
      "dürüst testte **gerçekten doğru çıkma oranıdır**.\n")
    w("| Model olasılığı (karar sınırından uzaklık) | Karar sayısı (120 video) | Gerçek doğruluk |\n|---|---|---|")
    for b in calib:
        acc = f"%{b['dogruluk'] * 100:.0f}" if b["dogruluk"] is not None else "–"
        w(f"| %{(0.5 + b['alt']) * 100:.0f} – %{min(100, (0.5 + b['ust']) * 100):.0f} | "
          f"{b['karar_sayisi']} | {acc} |")
    w("")
    w("Model kendinden daha emin olduğunda bile gerçek doğruluk belirgin biçimde artmıyor; bu yüzden "
      "sistem güveni abartmıyor, çoğu kararı \"düşük güven\" olarak işaretliyor.\n")
    w("---\n")
    w("## 7. Sistem nerede yanılıyor?\n")
    w(f"Yanlış karar verilen {len(wrong)} klip: " + ", ".join(
        f"**{c['klip']}** ({c['gercek'].lower()} → {c['karar'].lower()}, %{c['olasilik'] * 100:.0f})"
        for c in wrong) + ".\n")
    w("Bu hataların ortak nedenleri (geliştirme sırasında yapılan testlerle; ayrıntılar "
      "[YOL_HARITASI.md](YOL_HARITASI.md)):\n")
    w("1. **Temas anında iskelet noktaları en güvenilmez halinde.** Sporcular temas anında üst üste "
      "biniyor; poz modeli yüksek tekmede kafaya uzanan bacağı çoğu zaman yakalayamıyor (Bulgu 2). "
      "Daha büyük poz modeli de bunu çözmedi.")
    w("2. **İki boyutlu görüntüde \"başın hemen yanından geçen\" tekme ile \"başa değen\" tekme aynı "
      "görünüyor.** Ghost Hit kliplerinde de tekme başa çok yakın geçiyor; ayak-baş mesafesi bu yüzden "
      "ayırt edici değil (Bulgu 3).")
    w("3. **Başın tepkisi tek başına yetmiyor.** Ghost Hit'te de sporcular refleksle başını çeviriyor, "
      "geri çekiliyor ya da düşüyor; bu hareketler gerçek darbenin oluşturduğu dönüşe benziyor. "
      "Darbe karesi hakem tarafından işaretlense bile ayırt edicilik artmadı (Bulgu 7).")
    w("4. **Veri az.** 10 Real Hit ve 10 Ghost Hit klibiyle, %95 güven aralığı ±%20 civarında; "
      "küçük bir gerçek etkinin bile ölçülmesi güç.\n")
    w("---\n")
    w("## 8. Doğruluğun geliştirme boyunca değişimi\n")
    if has_hist:
        w("![Doğruluk geçmişi](sonuc/dogruluk_gecmisi.png)\n")
    w("Geliştirme sırasında ölçülen %45–%70 arasındaki değerler, kararın eşikleri veya modeli aynı 20 "
      "klip üzerinde ayarlandığı için **iyimserdi**. Dürüst teste (birini dışarıda bırakarak) geçildiğinde "
      f"doğruluk %{k_o / n_o * 100:.0f} çıktı. Bu fark, küçük veri setlerinde \"eğitildiği veride test "
      "etme\" hatasının ne kadar yanıltıcı olabileceğini gösteriyor.\n")
    w("---\n")
    w("## 9. Sonuç ve sonraki adımlar\n")
    w("- Sistem; video analizi, sporcu takibi, baş tepkisi ölçümü, kalibre güvenli karar, hakem girdileri "
      "ve PDF raporu ile **uçtan uca çalışan bir karar destek altyapısı** olarak tamamlandı.")
    w("- Mevcut 20 klipte, görüntüden çıkarılan ölçümler gerçek temas ile hayalet darbeyi **güvenilir "
      "biçimde ayıramıyor**. Bu, dürüst ölçümle ortaya konmuş bir bulgudur.")
    w("- **En önemli sonraki adım veri:** `Raw_Videos` klasöründeki 27 etiketlenmemiş olay etiketlenip "
      "eklendiğinde `python egit.py` ve `python sonuc_raporu.py` ile model ve bu rapor kod değişmeden "
      "yenilenebilir.")
    w("- Uzun vadede: birden fazla kamera açısı (üç boyutlu konum), temas anı için daha yüksek kare hızı "
      "ve dövüş sporlarına özel eğitilmiş poz modeli.")
    w("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"Doğruluk (20 orijinal): {k_o}/{n_o}  %95 GA: %{lo * 100:.0f}-%{hi * 100:.0f}  "
          f"şans p=%{p_chance * 100:.0f}")
    print(f"Doğruluk (120): {k_a}/{n_a}")
    print("Karmaşıklık (20):", used_cols, cm_o.tolist())
    print("Karmaşıklık (120):", cm_a.tolist())
    print("Yanlışlar:", [c["klip"] for c in wrong])
    print("Varyantlar:", by_var)
    print("AUC:", [(n, round(a, 2)) for n, _, _, a in feat_stats])
    print(f"Rapor: {os.path.relpath(OUT_MD, PROJECT_ROOT)}  Grafikler: {os.path.relpath(OUT_DIR, PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
