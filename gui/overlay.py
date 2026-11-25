"""
overlay.py — Video Üzerine Analiz İşaretlerini Çizme
=====================================================
Analiz sırasında her kare için kaydedilen baş ve ayak konumlarını, video
oynatılırken o anki kareye çizer. Böylece işaretler için videonun YOLO ile
ikinci kez işlenmesine ve tüm işaretli karelerin bellekte tutulmasına gerek kalmaz.

Analiz koordinatları "referans piksel" cinsindendir (bkz. segmentor.py, ölçek
normalizasyonu); çizimden önce ölçek faktörüne bölünerek videonun kendi
piksellerine çevrilir.
"""

import cv2

COLORS = {"red": (0, 0, 255), "blue": (255, 120, 0)}  # BGR
NOSE_COLOR = (0, 255, 0)
FOOT_RADIUS_PX = 7


def _pt(p, scale):
    return int(round(p[0] / scale)), int(round(p[1] / scale))


def draw_frame_features(frame, features: dict, scale: float, highlight: bool = False):
    """
    Bir kareye iki sporcunun baş bölgesini (daire), burnunu ve ayak noktalarını çizer.

    Parameters
    ----------
    frame : np.ndarray
        Videonun orijinal karesi (değiştirilmez, kopyası döner).
    features : dict
        main.analyze_video() sonucundaki "frame_features" listesinin bir elemanı.
    scale : float
        video_info["scale_factor"] (referans piksel / video pikseli).
    highlight : bool
        True ise (temas anı) karenin etrafına sarı çerçeve çizilir.
    """
    out = frame.copy()
    scale = scale or 1.0
    for color, bgr in COLORS.items():
        helmet = features.get(f"{color}_helmet", {})
        center, radius = helmet.get("head_center"), helmet.get("head_radius")
        if center is not None and radius:
            cv2.circle(out, _pt(center, scale), max(3, int(round(radius / scale))), bgr, 3)
        for nose in helmet.get("centroids", []):
            cv2.circle(out, _pt(nose, scale), 4, NOSE_COLOR, -1)
        foot = features.get(f"{color}_foot", {})
        for p in list(foot.get("centroids", [])) + list(foot.get("foot_tips", [])):
            cv2.circle(out, _pt(p, scale), FOOT_RADIUS_PX, bgr, -1)
    if highlight:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 255, 255), 8)
    return out
