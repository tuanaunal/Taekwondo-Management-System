"""
degerlendir.py — Sistemin Doğruluğunu Ölçen Değerlendirme Aracı
================================================================
Dataset klasöründeki videoları analiz eder, sistemin kararını videonun
gerçek etiketiyle karşılaştırır ve sonucu raporlar.

Etiketlerin anlamı (bkz. README "Problem Tanımı"):
    Real_Hit  -> kafaya gerçek temas VAR
    Ghost_Hit -> kafaya gerçek temas YOK

Kullanım:
    python degerlendir.py                    -> 20 orijinal video
    python degerlendir.py --tumu             -> 120 videonun tamamı (çoğaltılmışlar dahil)
    python degerlendir.py --not "adim 4"     -> sonuca bir açıklama notu ekler

Çıktılar (output/reports/):
    degerlendirme_<tarih>.csv     -> her videonun ayrıntılı sonucu
    degerlendirme_gecmisi.csv     -> her çalıştırmanın tek satırlık özeti;
                                     adımlar arasında karşılaştırma için
"""

import argparse
import csv
import io
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# Windows konsol encoding duzeltmesi
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import analyze_video
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR, REPORTS_DIR

# Karar motorunun (geçici) kategorilerinin "temas var / yok" karşılığı.
# INCONCLUSIVE (belirsiz) hiçbir sınıfa sayılmaz; her zaman yanlış kabul edilir.
CONTACT_DECISIONS = {"REAL_HIT", "LIGHT_CONTACT"}
NO_CONTACT_DECISIONS = {"GHOST_HIT", "EXTERNAL_FACTOR"}

AUGMENT_TAGS = ["flipped", "bright", "dark", "noisy", "blur"]
HISTORY_PATH = os.path.join(REPORTS_DIR, "degerlendirme_gecmisi.csv")
HISTORY_FIELDS = ["tarih", "kapsam", "video_sayisi", "dogru", "dogruluk",
                  "ghost_dogru", "ghost_toplam", "real_dogru", "real_toplam", "not"]


def video_variant(filename: str) -> str:
    """'GH_01_dark.mp4' -> 'dark', 'GH_01.mp4' -> 'orijinal'."""
    base = os.path.splitext(filename)[0]
    for tag in AUGMENT_TAGS:
        if base.endswith("_" + tag):
            return tag
    return "orijinal"


def clip_id(filename: str) -> str:
    """'GH_01_dark.mp4' -> 'GH_01' (bir klibin tüm kopyaları aynı kimliği taşır)."""
    return "_".join(os.path.splitext(filename)[0].split("_")[:2])


def collect_videos(include_augmented: bool) -> list:
    videos = []
    for label, directory in [("Ghost_Hit", GHOST_HIT_DIR), ("Real_Hit", REAL_HIT_DIR)]:
        if not os.path.exists(directory):
            print(f"  UYARI: Klasör bulunamadı: {directory}")
            continue
        for f in sorted(os.listdir(directory)):
            if not f.endswith(".mp4"):
                continue
            if not include_augmented and video_variant(f) != "orijinal":
                continue
            videos.append((label, os.path.join(directory, f)))
    return videos


def predicted_contact(decision: str):
    """Karardan 'temas var' (True), 'temas yok' (False) veya belirsiz (None) çıkarır."""
    if decision in CONTACT_DECISIONS:
        return True
    if decision in NO_CONTACT_DECISIONS:
        return False
    return None


def last_history_row(scope: str) -> dict:
    """Aynı kapsamdaki (orijinal / tumu) son çalıştırmanın özetini döner."""
    if not os.path.exists(HISTORY_PATH):
        return {}
    with open(HISTORY_PATH, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("kapsam") == scope]
    return rows[-1] if rows else {}


def main():
    parser = argparse.ArgumentParser(description="Sistemin doğruluğunu ölçer.")
    parser.add_argument("--tumu", action="store_true",
                        help="Çoğaltılmış videoları da dahil et (120 video).")
    parser.add_argument("--not", dest="note", default="",
                        help="Bu çalıştırma için kısa bir açıklama (ör. 'adim 4').")
    args = parser.parse_args()

    videos = collect_videos(args.tumu)
    scope = "tumu" if args.tumu else "orijinal"
    previous = last_history_row(scope)

    print("=" * 70)
    print(f"  DEĞERLENDİRME — {len(videos)} video ({scope})")
    print("=" * 70)

    rows = []
    start = time.time()

    for i, (label, path) in enumerate(videos, 1):
        name = os.path.basename(path)
        try:
            result = analyze_video(path, verbose=False, save_graph=False)
            dr = result["decision_result"]
            kr = result["kinematic_result"]
            cs = result["contact_summary"]
            decision = dr["decision"]
            details = {
                "hareket_tipi": kr.get("type", ""),
                "maks_ivme": round(float(kr.get("max_acceleration", 0.0)), 1),
                "net_savrulma": round(float(kr.get("net_displacement", 0.0)), 1),
                "temasli_kare": cs.get("contact_frames", 0),
                "min_mesafe": round(float(cs.get("min_distance", float("inf"))), 1),
                "gerekce": dr.get("reasoning", ""),
            }
        except Exception as e:  # bir videodaki hata tüm testi durdurmasın
            decision = "INCONCLUSIVE"
            details = {"gerekce": f"HATA: {e}"}

        pred = predicted_contact(decision)
        correct = pred is not None and pred == (label == "Real_Hit")
        rows.append({
            "video": name, "klip": clip_id(name), "varyasyon": video_variant(name),
            "gercek": label, "tahmin": decision, "dogru": int(correct), **details,
        })
        mark = "✓" if correct else "✗"
        print(f"  [{i:3d}/{len(videos)}] {mark} {name:22s} gerçek: {label:9s} → {decision}")

    if not rows:
        print("  Analiz edilecek video bulunamadı.")
        return

    ghost = [r for r in rows if r["gercek"] == "Ghost_Hit"]
    real = [r for r in rows if r["gercek"] == "Real_Hit"]
    ghost_ok = sum(r["dogru"] for r in ghost)
    real_ok = sum(r["dogru"] for r in real)
    total_ok = ghost_ok + real_ok
    accuracy = total_ok / len(rows)

    print("\n" + "─" * 70)
    print(f"  DOĞRULUK:   {total_ok}/{len(rows)}  (%{accuracy * 100:.1f})")
    print(f"  Ghost Hit:  {ghost_ok}/{len(ghost)}   (temas yok olan videolar)")
    print(f"  Real Hit:   {real_ok}/{len(real)}   (temas var olan videolar)")
    print(f"  Süre:       {time.time() - start:.0f} sn")

    def count(group, value):
        return sum(1 for r in group if predicted_contact(r["tahmin"]) is value)

    print("\n  Karmaşıklık matrisi       Tahmin: temas var | temas yok | belirsiz")
    print(f"    Gerçek: temas var (RH)        {count(real, True):4d}      | {count(real, False):4d}      | {count(real, None):4d}")
    print(f"    Gerçek: temas yok (GH)        {count(ghost, True):4d}      | {count(ghost, False):4d}      | {count(ghost, None):4d}")

    if args.tumu:
        # Dayanıklılık: aynı klibin farklı görüntü koşullarındaki kopyaları
        by_variant = defaultdict(list)
        by_clip = defaultdict(set)
        for r in rows:
            by_variant[r["varyasyon"]].append(r["dogru"])
            by_clip[r["klip"]].add(r["tahmin"])
        print("\n  Görüntü koşuluna göre doğruluk:")
        for v in ["orijinal"] + AUGMENT_TAGS:
            if by_variant[v]:
                print(f"    {v:9s} {sum(by_variant[v])}/{len(by_variant[v])}")
        stable = sum(1 for preds in by_clip.values() if len(preds) == 1)
        print(f"  Tüm kopyalarında aynı kararı alan klip: {stable}/{len(by_clip)}")

    if previous:
        print(f"\n  Önceki çalıştırma ({previous['tarih']}, not: {previous.get('not') or '-'}): "
              f"{previous['dogru']}/{previous['video_sayisi']} (%{float(previous['dogruluk']) * 100:.1f})")

    # ── Kayıt ──
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    detail_path = os.path.join(REPORTS_DIR, f"degerlendirme_{stamp}_{scope}.csv")
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    new_file = not os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"), "kapsam": scope,
            "video_sayisi": len(rows), "dogru": total_ok,
            "dogruluk": round(accuracy, 4),
            "ghost_dogru": ghost_ok, "ghost_toplam": len(ghost),
            "real_dogru": real_ok, "real_toplam": len(real), "not": args.note,
        })

    print(f"\n  Ayrıntılı sonuç: {detail_path}")
    print(f"  Geçmiş:          {HISTORY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
