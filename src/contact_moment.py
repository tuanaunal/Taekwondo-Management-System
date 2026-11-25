"""
contact_moment.py — Tartışmalı Anın (Temas Anı) Bulunması
==========================================================
Bir klipteki analizin referans noktası olan "temas anını" bulur.
Sonraki adımlar (temas ölçümü, baş hareketi analizi) bu anın
öncesine ve sonrasına odaklanır.

Anın belirlenme sırası:
  1. Kullanıcı (ör. hakem) sensörün puan verdiği kareyi verdiyse o kare kullanılır.
  2. Saldıranın ayağı rakibin başına yeterince yaklaştıysa, en yakın yaklaşım karesi.
  3. Yaklaşım görülemediyse (iskelet modeli tekme atan bacağı kaçırabiliyor),
     sporculardan birinin başının en hızlı hareket ettiği kare (darbe belirtisi).

Tüm mesafeler "referans piksel" cinsindendir (26 px = bir baş yarıçapı, bkz. segmentor.py).
"""

import numpy as np

# Ayak, rakibin baş merkezine bu kadar baş yarıçapı yakınsa "yaklaşım" güvenilir kabul edilir.
# (Baş yarıçapı + ayak ucu tahmin hatası payı: 3 baş yarıçapı ~ 25 cm)
CLOSE_APPROACH_RADII = 3.0

METHOD_USER = "kullanici"
METHOD_APPROACH = "en_yakin_yaklasim"
METHOD_HEAD_MOTION = "bas_hareketi"
METHOD_NONE = "bulunamadi"

METHOD_LABELS_TR = {
    METHOD_USER: "Kullanıcının işaretlediği kare",
    METHOD_APPROACH: "Ayağın başa en çok yaklaştığı kare",
    METHOD_HEAD_MOTION: "Başın en hızlı hareket ettiği kare",
    METHOD_NONE: "Temas anı bulunamadı",
}


def frames_to_ranges(frames) -> list:
    """İşaretli kareleri ardışık aralıklara çevirir, ör. [10, 11, 12, 20] -> [(10, 12), (20, 20)]."""
    ranges = []
    for f in sorted(set(int(x) for x in frames)):
        if ranges and f == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], f)
        else:
            ranges.append((f, f))
    return ranges


def _distance(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def closest_approach_profile(frame_features: list) -> list:
    """
    Her kare için saldıran ayak ile rakip baş arasındaki en kısa mesafeyi
    (baş yarıçapı cinsinden) ve hangi başın hedef olduğunu döner.

    frame_features: her kare için segmentor.build_features() çıktısı
    Dönüş: [(mesafe_yaricap veya inf, hedef_renk veya None), ...]
    """
    profile = []
    for seg in frame_features:
        best, target = np.inf, None
        for foot_key, head_key, head_color in [("red_foot", "blue_helmet", "blue"),
                                               ("blue_foot", "red_helmet", "red")]:
            head = seg.get(head_key, {})
            center, radius = head.get("head_center"), head.get("head_radius")
            if center is None or not radius:
                continue
            feet = seg.get(foot_key, {})
            points = list(feet.get("foot_tips", [])) + list(feet.get("centroids", []))
            for pt in points:
                d = _distance(pt, center) / radius
                if d < best:
                    best, target = d, head_color
        profile.append((best, target))
    return profile


def head_speed_profile(frame_features: list, fps: float) -> dict:
    """
    Her sporcunun başının (burun noktası) kare kare hızını döner (referans px/sn).

    Sahte hız tepelerini önlemek için şu karelerde hız NaN kabul edilir:
      - baş görünmüyorsa,
      - baş konumu ölçülmeyip son konumdan tutulduysa (tutma bitince konum sıçrar),
      - sporcunun takip numarası değiştiyse (kimlik geçişinde nokta başka kişiye atlayabilir).
    """
    speeds = {}
    for color in ("red", "blue"):
        pts, ids, held = [], [], []
        for seg in frame_features:
            helmet = seg.get(f"{color}_helmet", {})
            c = helmet.get("centroids", [])
            pts.append(c[0] if c else None)
            ids.append(helmet.get("track_id"))
            held.append(bool(helmet.get("held", False)))
        v = np.full(len(pts), np.nan)
        for i in range(1, len(pts)):
            if pts[i] is None or pts[i - 1] is None:
                continue
            if held[i] or held[i - 1] or ids[i] != ids[i - 1]:
                continue
            v[i] = _distance(pts[i], pts[i - 1]) * fps
        speeds[color] = v
    return speeds


def find_contact_moment(frame_features: list, fps: float, user_frame=None) -> dict:
    """
    Temas anını bulur.

    Returns
    -------
    dict
        {
            'frame': int veya None,      # temas anının kare numarası
            'time_sec': float veya None,
            'method': str,               # kullanici / en_yakin_yaklasim / bas_hareketi / bulunamadi
            'method_tr': str,
            'target': 'red' | 'blue' | None,   # başı hedef alınan sporcu
            'approach_radii': float,     # o andaki ayak-baş mesafesi (baş yarıçapı cinsinden)
            'approach_profile': list,    # her kare için ayak-baş mesafesi (yarıçap)
        }
    """
    n = len(frame_features)
    approach = closest_approach_profile(frame_features)
    distances = np.array([d for d, _ in approach], dtype=float)
    result = {
        "frame": None, "time_sec": None, "method": METHOD_NONE,
        "target": None, "approach_radii": float("inf"),
        "approach_profile": [float(d) for d in distances],
    }

    frame, method, target = None, METHOD_NONE, None
    if user_frame is not None and 0 <= int(user_frame) < n:
        frame, method = int(user_frame), METHOD_USER
        target = approach[frame][1]
    elif n > 0 and np.isfinite(distances).any() and np.nanmin(distances) <= CLOSE_APPROACH_RADII:
        frame = int(np.nanargmin(np.where(np.isfinite(distances), distances, np.nan)))
        method, target = METHOD_APPROACH, approach[frame][1]
    elif n > 1:
        speeds = head_speed_profile(frame_features, fps)
        peaks = {c: (np.nanmax(v) if np.isfinite(v).any() else -1.0) for c, v in speeds.items()}
        color = max(peaks, key=peaks.get)
        if peaks[color] > 0:
            frame = int(np.nanargmax(speeds[color]))
            method, target = METHOD_HEAD_MOTION, color

    if frame is not None:
        result.update({
            "frame": frame,
            "time_sec": frame / fps if fps else None,
            "method": method,
            "target": target,
            "approach_radii": float(distances[frame]),
        })
    result["method_tr"] = METHOD_LABELS_TR[result["method"]]
    return result
