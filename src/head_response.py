"""
head_response.py — Başın Darbeye Verdiği Tepki
===============================================
Projenin ana ayırt edici ölçümleri. İki boyutlu görüntüde temas, temassız
geçişten ayırt edilemediği için (YOL_HARITASI.md, Bulgu 3) karar, başın
fiziksel tepkisine dayanır:

1. Başın gövdeye göre savrulması
   Sporcu geri kaçarken baş ve gövde birlikte hareket eder; darbe aldığında
   baş, gövdeye göre ani bir hareket yapar. Baş merkezi ile gövde merkezi
   arasındaki vektörün değişimi ölçülür (gövdenin kendi hareketi çıkarılmış olur).

2. Baş dönüşü
   Darbe başı döndürür (Bulgu 1: Real Hit'te burun, baş merkezinden ~%35 fazla
   hareket ediyor). Burnun baş merkezine göre konumu (baş yarıçapı cinsinden)
   başın yönünü gösterir; bu vektörün değişimi dönüş miktarıdır.

Her ölçüm, bir kare penceresinin (±0.33 sn) başı ile sonu arasındaki değişimdir.
Temas anı verilirse pencere merkezi temas anının ±0.17 sn çevresinde aranır;
verilmezse klibin tamamındaki en büyük değer alınır. Tüm mesafeler referans
pikseldir (26 px = bir baş yarıçapı, bkz. segmentor.py).
"""

import numpy as np
from scipy.signal import savgol_filter

WINDOW_HALF_SEC = 0.33     # Değişimin ölçüldüğü pencerenin yarı süresi (30 FPS'te 10 kare)
SMOOTH_WINDOW_SEC = 0.2    # Savitzky–Golay yumuşatma penceresi (kinematics.py ile aynı)
MAX_GAP_SEC = 0.17         # Bu süreden kısa eksiklikler ara değerle doldurulur (30 FPS'te 5 kare)
ANCHOR_TOLERANCE_SEC = 0.17  # Temas anı tahmini ± bu kadar kayabilir (30 FPS'te 5 kare)
RANGE_TOLERANCE_SEC = 0.07   # Kullanıcının verdiği kare aralıkları ± bu kadar genişletilir (30 FPS'te 2 kare)


def _fill_and_smooth(series: np.ndarray, fps: float) -> np.ndarray:
    """(n, 2) seride kısa boşlukları doğrusal ara değerle doldurur ve Savitzky–Golay ile yumuşatır."""
    out = series.copy()
    n = len(out)
    valid = ~np.isnan(out[:, 0])
    if valid.sum() < 2:
        return out
    idx = np.arange(n)
    max_gap = max(1, int(round(MAX_GAP_SEC * fps)))
    # Kısa boşlukları doldur (uzun boşluklar NaN kalır)
    vi = idx[valid]
    for a, b in zip(vi[:-1], vi[1:]):
        if 1 < b - a <= max_gap + 1:
            for d in range(2):
                out[a + 1:b, d] = np.interp(idx[a + 1:b], [a, b], [out[a, d], out[b, d]])
    # Kesintisiz parçaları ayrı ayrı yumuşat
    w = int(round(SMOOTH_WINDOW_SEC * fps))
    w = w + 1 if w % 2 == 0 else w
    valid = ~np.isnan(out[:, 0])
    start = None
    for i in range(n + 1):
        if i < n and valid[i]:
            start = i if start is None else start
        elif start is not None:
            seg = out[start:i]
            ww = min(w, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
            if ww >= 3:
                out[start:i] = savgol_filter(seg, ww, 2, axis=0, mode="interp")
            start = None
    return out


def _max_window_change(series: np.ndarray, half: int, centers=None):
    """
    Pencere başı (t-half) ile sonu (t+half) arasındaki değişimin en büyüğü ve merkez karesi.
    centers verilirse sadece bu merkez kareler denenir (temas anına odaklanma).
    """
    best, best_t = 0.0, None
    candidates = range(half, len(series) - half) if centers is None else \
        [t for t in centers if half <= t < len(series) - half]
    for t in candidates:
        a, b = series[t - half], series[t + half]
        if np.isnan(a[0]) or np.isnan(b[0]):
            continue
        d = float(np.linalg.norm(b - a))
        if d > best:
            best, best_t = d, t
    return best, best_t


def athlete_series(frame_features: list, color: str) -> dict:
    """Bir sporcunun kare kare burun, baş merkezi, baş yarıçapı ve gövde merkezi serilerini çıkarır."""
    n = len(frame_features)
    nose = np.full((n, 2), np.nan)
    head = np.full((n, 2), np.nan)
    torso = np.full((n, 2), np.nan)
    radius = np.full(n, np.nan)
    for i, seg in enumerate(frame_features):
        h = seg.get(f"{color}_helmet", {})
        if h.get("held"):
            continue  # Bu karede ölçülmemiş (son konum tutulmuş): veri yok say
        c = h.get("centroids", [])
        if c:
            nose[i] = c[0]
        if h.get("head_center") is not None:
            head[i] = h["head_center"]
        if h.get("torso_center") is not None:
            torso[i] = h["torso_center"]
        if h.get("head_radius"):
            radius[i] = h["head_radius"]
    return {"nose": nose, "head": head, "torso": torso, "radius": radius}


def compute_head_response(frame_features: list, fps: float,
                          anchor_frame: int = None, target: str = None,
                          anchor_frames=None) -> dict:
    """
    Her iki sporcu için baş tepkisi ölçümlerini hesaplar.

    anchor_frame: temas anı (verilirse ölçümler bu anın ±0.17 sn çevresinden alınır)
    anchor_frames: kullanıcının işaretlediği vuruş karelerinin tamamı (bir veya birden çok aralık).
        Verilirse ölçümler bu karelerin (±0.07 sn) çevresinden alınır ve aday anlar arasından
        EN BÜYÜK baş tepkisi kullanılır: gerçek darbe varsa en güçlü tepkiyi o verir.
    target: başı hedef alınan sporcu ('red' / 'blue'); verilirse 'max' onun değerleridir

    Returns
    -------
    dict
        {
          'red' / 'blue': {
              'absolute_swing':   burnun ekrandaki savrulması (referans px) — eski ölçüm, karşılaştırma için
              'relative_swing':   baş merkezinin gövdeye göre savrulması (referans px)
              'rotation':         baş dönüşü (burnun baş merkezine göre kayması, baş yarıçapı cinsinden)
              '*_frame':          ilgili ölçümün en büyük olduğu kare
          },
          'max': hedef sporcunun değerleri; hedef bilinmiyorsa iki sporcunun en büyükleri,
          'anchor_frame', 'target': kullanılan temas anı ve hedef
        }
    """
    half = max(1, int(round(WINDOW_HALF_SEC * fps)))
    centers = None
    if anchor_frames:
        tol = max(0, int(round(RANGE_TOLERANCE_SEC * fps)))
        centers = sorted({int(f) + d for f in anchor_frames for d in range(-tol, tol + 1)})
    elif anchor_frame is not None:
        tol = max(0, int(round(ANCHOR_TOLERANCE_SEC * fps)))
        centers = range(int(anchor_frame) - tol, int(anchor_frame) + tol + 1)
    result = {}
    for color in ("red", "blue"):
        s = athlete_series(frame_features, color)
        nose = _fill_and_smooth(s["nose"], fps)
        head = _fill_and_smooth(s["head"], fps)
        torso = _fill_and_smooth(s["torso"], fps)
        r = np.nanmedian(s["radius"]) if np.isfinite(s["radius"]).any() else np.nan

        relative = head - torso                 # gövdeye göre baş konumu
        orientation = (nose - head) / r if np.isfinite(r) and r > 0 else np.full_like(nose, np.nan)

        n_frames = len(frame_features)
        coverage = float(np.count_nonzero(~np.isnan(s["nose"][:, 0]))) / n_frames if n_frames else 0.0
        abs_v, abs_t = _max_window_change(nose, half, centers)
        rel_v, rel_t = _max_window_change(relative, half, centers)
        rot_v, rot_t = _max_window_change(orientation, half, centers)
        result[color] = {
            "absolute_swing": abs_v, "absolute_frame": abs_t,
            "relative_swing": rel_v, "relative_frame": rel_t,
            "rotation": rot_v, "rotation_frame": rot_t,
            "coverage": coverage,  # başın gerçekten ölçüldüğü karelerin oranı
        }
    keys = ("absolute_swing", "relative_swing", "rotation")
    if target in ("red", "blue"):
        result["max"] = {key: result[target][key] for key in keys}
        result["coverage"] = result[target]["coverage"]
    else:
        result["max"] = {key: max(result["red"][key], result["blue"][key]) for key in keys}
        # Hedef bilinmiyorsa: iki sporcudan daha iyi ölçülenin kapsaması
        result["coverage"] = max(result["red"]["coverage"], result["blue"]["coverage"])
    result["anchor_frame"] = anchor_frame
    result["target"] = target
    return result
