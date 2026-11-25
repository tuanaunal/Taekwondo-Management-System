"""
egit.py — Karar Modelinin Eğitimi ve Dürüst Değerlendirmesi
============================================================
Karar modeli, başın darbeye verdiği tepkiyi (baş dönüşü, burun savrulması) ve
ayak-baş mesafesini "temas var" olasılığına çeviren küçük bir lojistik
regresyondur. Bu araç:

  1. Veri setindeki tüm videolardan (çoğaltılmışlar dahil) ölçümleri çıkarır.
  2. Modeli "birini dışarıda bırakarak" (leave-one-out) değerlendirir:
     her seferinde BİR KLİP ve onun TÜM ÇOĞALTILMIŞ KOPYALARI dışarıda tutulur,
     model kalan kliplerle eğitilir ve dışarıdaki klipte test edilir.
     Böylece ölçülen doğruluk, sistemin hiç görmediği bir klipteki başarısını
     gösterir (eşikler test edilen veriye göre ayarlanmaz).
  3. Son modeli tüm verilerle eğitip model/karar_modeli.json dosyasına kaydeder.

Kullanım:
    python egit.py               -> ölçümleri yeniden çıkarır (~10-15 dk), değerlendirir, kaydeder
    python egit.py --onbellek    -> önceki ölçümleri kullanır (kod değişmediyse, hızlı)
"""

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from src.config import GHOST_HIT_DIR, REAL_HIT_DIR, REPORTS_DIR, PROJECT_ROOT
from src.decision_engine import FEATURES, FEATURE_LABELS_TR, MODEL_PATH, extract_features

CACHE_PATH = os.path.join(REPORTS_DIR, "ozellikler.csv")
REGULARIZATION_C = 1.0   # Küçük veri setinde aşırı uyumu sınırlamak için L2 düzenlileştirme
DECISION_THRESHOLD = 0.5
# Güven kalibrasyonu: modelin olasılığının karar sınırına (0.5) uzaklığına göre gruplar.
# Her grupta, birini dışarıda bırakarak yapılan tahminlerin GERÇEK doğruluk oranı ölçülür;
# arayüzde gösterilen güven bu orandır (modelin kendi iddiası değil).
CALIBRATION_BINS = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.51)]
AUGMENT_TAGS = ["flipped", "bright", "dark", "noisy", "blur"]


def clip_id(filename: str) -> str:
    return "_".join(os.path.splitext(filename)[0].split("_")[:2])


def is_original(filename: str) -> bool:
    base = os.path.splitext(filename)[0]
    return not any(base.endswith("_" + t) for t in AUGMENT_TAGS)


def extract_all() -> list:
    from main import analyze_video
    rows = []
    videos = []
    for label, d in [("Ghost_Hit", GHOST_HIT_DIR), ("Real_Hit", REAL_HIT_DIR)]:
        videos += [(label, os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".mp4")]
    for i, (label, path) in enumerate(videos, 1):
        name = os.path.basename(path)
        r = analyze_video(path, verbose=False, save_graph=False)
        feats = extract_features(r["contact_summary"], r["head_response"])
        row = {"video": name, "klip": clip_id(name), "orijinal": int(is_original(name)),
               "etiket": 1 if label == "Real_Hit" else 0,
               "kapsama": round(r["head_response"].get("coverage", 1.0), 3), **feats}
        rows.append(row)
        print(f"  [{i:3d}/{len(videos)}] {name:22s} " +
              "  ".join(f"{FEATURE_LABELS_TR[k]}={feats[k]:.2f}" for k in FEATURES), flush=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def load_cache() -> list:
    with open(CACHE_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FEATURES + ["kapsama"]:
            r[k] = float(r[k])
        r["etiket"], r["orijinal"] = int(r["etiket"]), int(r["orijinal"])
    return rows


def fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Ölçümleri standartlaştırıp lojistik regresyon eğitir (karar motorunun okuduğu biçimde)."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=REGULARIZATION_C)
    clf.fit((X - mean) / std, y)
    return {"mean": mean.tolist(), "std": std.tolist(),
            "weights": clf.coef_[0].tolist(), "bias": float(clf.intercept_[0])}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    z = (X - np.array(model["mean"])) / np.array(model["std"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(model["weights"]) + model["bias"])))


def leave_one_clip_out(rows: list, features: list) -> np.ndarray:
    """Her klip (tüm kopyalarıyla) bir kez dışarıda tutulur; dışarıdaki için tahmin üretilir."""
    clips = sorted(set(r["klip"] for r in rows))
    X = np.array([[r[k] for k in features] for r in rows])
    y = np.array([r["etiket"] for r in rows])
    probs = np.zeros(len(rows))
    for c in clips:
        test = np.array([r["klip"] == c for r in rows])
        m = fit(X[~test], y[~test])
        probs[test] = predict(m, X[test])
    return probs


def calibration_table(rows: list, probs: np.ndarray) -> list:
    """Birini dışarıda bırakarak elde edilen olasılıklardan güven kalibrasyon tablosu."""
    y = np.array([r["etiket"] for r in rows])
    pred = (probs >= DECISION_THRESHOLD).astype(int)
    dist = np.abs(probs - DECISION_THRESHOLD)
    table = []
    for lo, hi in CALIBRATION_BINS:
        m = (dist >= lo) & (dist < hi)
        acc = float((pred[m] == y[m]).mean()) if m.any() else None
        table.append({"alt": lo, "ust": hi, "karar_sayisi": int(m.sum()),
                      "dogruluk": round(acc, 4) if acc is not None else None})
    return table


def report(rows: list, probs: np.ndarray, title: str):
    y = np.array([r["etiket"] for r in rows])
    pred = (probs >= DECISION_THRESHOLD).astype(int)
    orig = np.array([r["orijinal"] == 1 for r in rows])
    acc_o = (pred[orig] == y[orig]).mean()
    acc_a = (pred == y).mean()
    rh = orig & (y == 1)
    gh = orig & (y == 0)
    print(f"  {title:44s} orijinal 20: {int((pred[orig] == y[orig]).sum())}/{orig.sum()} (%{acc_o * 100:.0f})"
          f"  [Real {int((pred[rh] == 1).sum())}/{rh.sum()}, Ghost {int((pred[gh] == 0).sum())}/{gh.sum()}]"
          f"   tüm 120: %{acc_a * 100:.0f}")
    return acc_o, acc_a


def main():
    parser = argparse.ArgumentParser(description="Karar modelini eğitir ve değerlendirir.")
    parser.add_argument("--onbellek", action="store_true", help="Önceki ölçümleri kullan.")
    args = parser.parse_args()

    print("=" * 78)
    print("  KARAR MODELİ EĞİTİMİ")
    print("=" * 78)
    if args.onbellek and os.path.exists(CACHE_PATH):
        rows = load_cache()
        print(f"  Önbellekten {len(rows)} video ölçümü yüklendi: {CACHE_PATH}")
    else:
        print("  Ölçümler çıkarılıyor...")
        rows = extract_all()

    print("\n  Birini dışarıda bırakarak doğrulama (her klip tüm kopyalarıyla bir kez test):")
    results = {}
    loo_probs = None
    for name, feats in [("Sadece baş dönüşü", ["rotation"]),
                        ("Sadece burun savrulması", ["absolute_swing"]),
                        ("Baş dönüşü + burun savrulması", ["rotation", "absolute_swing"]),
                        ("Tüm ölçümler (+ ayak-baş mesafesi)", FEATURES)]:
        probs = leave_one_clip_out(rows, feats)
        results[name] = report(rows, probs, name)
        if feats == FEATURES:
            loo_probs = probs

    # Son model: tüm ölçümler, tüm veri
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["etiket"] for r in rows])
    model = {"features": FEATURES, **fit(X, y), "threshold": DECISION_THRESHOLD,
             "kalibrasyon": calibration_table(rows, loo_probs)}
    loo_o, loo_a = results["Tüm ölçümler (+ ayak-baş mesafesi)"]
    model.update({
        "egitim_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "egitim_video_sayisi": int(len(rows)),
        "dogrulama": {"yontem": "birini dışarıda bırakarak (klip + tüm kopyaları)",
                      "orijinal_20_dogruluk": round(float(loo_o), 4),
                      "tum_120_dogruluk": round(float(loo_a), 4)},
    })
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print("\n  Güven kalibrasyonu (model olasılığı -> birini dışarıda bırakarak gerçek doğruluk):")
    for b in model["kalibrasyon"]:
        acc = f"%{b['dogruluk'] * 100:.0f}" if b["dogruluk"] is not None else "-"
        print(f"    olasılık %{(0.5 + b['alt']) * 100:.0f}-%{min(100, (0.5 + b['ust']) * 100):.0f}: "
              f"{b['karar_sayisi']:3d} karar, gerçek doğruluk {acc}")

    print("\n  Son model (tüm veriyle eğitildi) ağırlıkları — pozitif = temas lehine:")
    for k, w in zip(FEATURES, model["weights"]):
        print(f"    {FEATURE_LABELS_TR[k]:20s} {w:+.2f}")
    print(f"\n  Model kaydedildi: {os.path.relpath(MODEL_PATH, PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
