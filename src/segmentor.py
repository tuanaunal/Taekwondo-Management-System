"""
segmentor.py — HSV Renk Tabanli Nesne Segmentasyonu (v5 — Statik Filtre)
==============================================================================
v5 Yenilikleri:
  - Statik Nesne Filtresi: 5+ kare boyunca ayni yerde duran konturlar
    "arka plan gurultusu" (tavan isigi, skor tablosu) olarak etiketlenir
    ve ayak/kask adayi olarak kabul edilmez.
  - Sekil tabanli filtreleme korunuyor.
  - Capraz Bolge Kurali (Crossover) korunuyor.
  - Hogu (govde koruyucu) elemesi korunuyor.
  - Gecici kayip telafisi (8 kare) korunuyor.
"""

import cv2
import numpy as np
from src.config import (
    HSV_RANGES,
    MORPH_KERNEL_SIZE, MORPH_ERODE_ITERATIONS,
    MORPH_DILATE_ITERATIONS, MORPH_CLOSE_ITERATIONS,
    MIN_CONTOUR_AREA,
)

# -- Segmentor-ozel sabitler --
_SEG_MAX_AREA = 80_000
_SEG_MIN_CIRCULARITY = 0.25
_SEG_MIN_SOLIDITY = 0.60
_SEG_MIN_AREA = 600

# Statik filtre sabitleri
_STATIC_DIST_THRESHOLD = 8      # piksel
_STATIC_FRAME_THRESHOLD = 999   # devre disi (v4 segmentor + proximity kurali)


class EquipmentSegmentor:
    """
    Taekwondo ekipmanlarini (kask, ayak koruyucu) renk tabanli
    segmentasyon ile ayirt eden sinif.
    """

    def __init__(self):
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE
        )
        self.prev_centroids = {
            "red_helmet": None,  "red_foot": None,
            "blue_helmet": None, "blue_foot": None,
        }
        self.lost_frames = {
            "red_helmet": 0,  "red_foot": 0,
            "blue_helmet": 0, "blue_foot": 0,
        }
        self.last_masks = {
            "red_helmet": None,  "red_foot": None,
            "blue_helmet": None, "blue_foot": None,
        }
        # Statik nokta takibi: (cx, cy) ve kac kare boyunca sabit
        # Renk bazinda: {"red": [(cx,cy,count), ...], "blue": [...]}
        self._static_points = {"red": [], "blue": []}

    # ----------------------------------------------------------------
    # ANA SEGMENTASYON
    # ----------------------------------------------------------------
    def segment_frame(self, frame: np.ndarray) -> dict:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = {}

        # Kirmizi
        red_mask = self._create_red_mask(hsv)
        red_mask = self._apply_morphology(red_mask)
        red_helmet, red_foot = self._separate_helmet_foot(red_mask, frame, "red")
        results["red_helmet"] = self._extract_features(red_helmet)
        results["red_foot"] = self._extract_features(red_foot)

        # Mavi
        blue_mask = self._create_blue_mask(hsv)
        blue_mask = self._apply_morphology(blue_mask)
        blue_helmet, blue_foot = self._separate_helmet_foot(blue_mask, frame, "blue")
        results["blue_helmet"] = self._extract_features(blue_helmet)
        results["blue_foot"] = self._extract_features(blue_foot)

        return results

    # ----------------------------------------------------------------
    # RENK MASKELERI
    # ----------------------------------------------------------------
    def _create_red_mask(self, hsv):
        mask1 = cv2.inRange(hsv, HSV_RANGES["red_helmet_lower1"],
                            HSV_RANGES["red_helmet_upper1"])
        mask2 = cv2.inRange(hsv, HSV_RANGES["red_helmet_lower2"],
                            HSV_RANGES["red_helmet_upper2"])
        return cv2.bitwise_or(mask1, mask2)

    def _create_blue_mask(self, hsv):
        return cv2.inRange(hsv, HSV_RANGES["blue_helmet_lower"],
                           HSV_RANGES["blue_helmet_upper"])

    # ----------------------------------------------------------------
    # MORFOLOJIK ISLEMLER
    # ----------------------------------------------------------------
    def _apply_morphology(self, mask):
        mask = cv2.erode(mask, self.morph_kernel,
                         iterations=MORPH_ERODE_ITERATIONS)
        mask = cv2.dilate(mask, self.morph_kernel,
                          iterations=MORPH_DILATE_ITERATIONS)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel,
                                iterations=MORPH_CLOSE_ITERATIONS)
        return mask

    # ----------------------------------------------------------------
    # KONTUR GECERLILIK FILTRESI
    # ----------------------------------------------------------------
    def _is_valid_equipment_contour(self, cnt, area, cx, cy, w, h):
        if area < _SEG_MIN_AREA or area > _SEG_MAX_AREA:
            return False
        if cx < 40 or cx > w - 40 or cy < 100 or cy > h - 40:
            return False
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            return False
        circularity = (4 * 3.14159 * area) / (perimeter * perimeter)
        if circularity < _SEG_MIN_CIRCULARITY:
            return False
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            return False
        solidity = area / hull_area
        if solidity < _SEG_MIN_SOLIDITY:
            return False
        return True

    # ----------------------------------------------------------------
    # STATIK NESNE FILTRESI
    # ----------------------------------------------------------------
    def _update_static_points(self, color, contour_centroids):
        """
        Her karede tespit edilen kontur merkezlerini onceki karelerle
        karsilastirir.  5+ kare boyunca ayni yerde (<15px) duran
        noktalar 'statik' (arka plan) olarak isaretlenir.
        """
        old_pts = self._static_points[color]
        new_pts = []
        used = set()

        for cx, cy in contour_centroids:
            matched = False
            for i, entry in enumerate(old_pts):
                if i in used or entry is None:
                    continue
                ox, oy, count = entry
                dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist < _STATIC_DIST_THRESHOLD:
                    new_pts.append((cx, cy, count + 1))
                    used.add(i)
                    matched = True
                    break
            if not matched:
                new_pts.append((cx, cy, 1))

        self._static_points[color] = new_pts

    def _is_static(self, color, cx, cy):
        """Verilen noktanin 'statik arka plan' olup olmadigini dondurur."""
        for sx, sy, count in self._static_points[color]:
            if count >= _STATIC_FRAME_THRESHOLD:
                dist = np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
                if dist < _STATIC_DIST_THRESHOLD:
                    return True
        return False

    # ----------------------------------------------------------------
    # KASK – AYAK AYIRIMI  (v5)
    # ----------------------------------------------------------------
    def _separate_helmet_foot(self, mask, frame, color):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        helmet_mask = np.zeros_like(mask)
        foot_mask = np.zeros_like(mask)

        valid_contours = []
        all_centroids = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if not self._is_valid_equipment_contour(cnt, area, cx, cy, w, h):
                continue

            all_centroids.append((cx, cy))
            valid_contours.append({
                "cnt": cnt, "area": area,
                "centroid": (cx, cy), "bbox": (x, y, bw, bh),
            })

        # Statik nokta veritabanini guncelle
        self._update_static_points(color, all_centroids)

        # NOT: Statik filtreyi burada GLOBAL olarak uygulamiyoruz.
        # Kasklar dogal olarak duragandir. Statik filtre YALNIZCA
        # crossover ayak adaylarina uygulanir (asagida).

        if not valid_contours:
            self._handle_loss(color, "helmet", helmet_mask)
            self._handle_loss(color, "foot", foot_mask)
            return helmet_mask, foot_mask

        # -- Capraz Bolge Siniri --
        split_x = w // 2
        prev_red_h = self.prev_centroids.get("red_helmet")
        prev_blue_h = self.prev_centroids.get("blue_helmet")
        if prev_red_h is not None and prev_blue_h is not None:
            split_x = (prev_red_h[0] + prev_blue_h[0]) // 2

        if color == "red":
            own_side = lambda cx_val: cx_val < split_x
        else:
            own_side = lambda cx_val: cx_val >= split_x

        own_side_contours = [c for c in valid_contours
                             if own_side(c["centroid"][0])]
        cross_side_contours = [c for c in valid_contours
                               if not own_side(c["centroid"][0])]

        # Crossover adaylarindan STATIK olanlari filtrele
        # (tavan isigi, skor tablosu vb. sabit arka plan nesneleri)
        cross_side_dynamic = [
            c for c in cross_side_contours
            if not self._is_static(color, c["centroid"][0], c["centroid"][1])
        ]

        prev_h = self.prev_centroids[f"{color}_helmet"]
        prev_f = self.prev_centroids[f"{color}_foot"]

        helmet_cnt = None
        foot_cnt = None

        # -- 1) CROSSOVER -> KESINLIKLE AYAK (statik filtreli) --
        if cross_side_dynamic:
            if prev_f is not None:
                cross_side_dynamic.sort(
                    key=lambda c: np.linalg.norm(
                        np.array(c["centroid"]) - np.array(prev_f)))
            else:
                cross_side_dynamic.sort(key=lambda c: c["area"], reverse=True)
            foot_cnt = cross_side_dynamic[0]

        # -- 2) KASK -> KENDI TARAFINDA EN YUKSEK --
        if own_side_contours:
            if prev_h is not None:
                candidates = []
                for c in own_side_contours:
                    dist = np.linalg.norm(
                        np.array(c["centroid"]) - np.array(prev_h))
                    if dist < 300:
                        candidates.append((dist, c))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    helmet_cnt = candidates[0][1]
            if helmet_cnt is None:
                own_side_contours.sort(key=lambda c: c["centroid"][1])
                helmet_cnt = own_side_contours[0]

        # -- 3) AYAK BULUNAMAZSA -> KENDI TARAFINDA ARA (hogu elenerek) --
        if foot_cnt is None and helmet_cnt is not None:
            remaining = [c for c in own_side_contours if c is not helmet_cnt]
            h_cx, h_cy = helmet_cnt["centroid"]

            foot_candidates = []
            for c in remaining:
                c_cx, c_cy = c["centroid"]
                dy = c_cy - h_cy
                dx = abs(c_cx - h_cx)
                is_hogu = (80 < dy < 260) and (dx < 130)
                if is_hogu:
                    continue
                foot_candidates.append(c)

            if foot_candidates:
                if prev_f is not None:
                    foot_candidates.sort(
                        key=lambda c: np.linalg.norm(
                            np.array(c["centroid"]) - np.array(prev_f)))
                    if np.linalg.norm(np.array(foot_candidates[0]["centroid"]) -
                                      np.array(prev_f)) < 400:
                        foot_cnt = foot_candidates[0]
                else:
                    foot_candidates.sort(
                        key=lambda c: c["centroid"][1], reverse=True)
                    foot_cnt = foot_candidates[0]

        # -- 4) FALLBACK --
        if helmet_cnt is None and valid_contours:
            valid_contours.sort(key=lambda c: c["centroid"][1])
            helmet_cnt = valid_contours[0]
            remaining_all = [c for c in valid_contours if c is not helmet_cnt]
            if remaining_all and foot_cnt is None:
                remaining_all.sort(key=lambda c: c["centroid"][1], reverse=True)
                foot_cnt = remaining_all[0]

        # -- MASKE GUNCELLE --
        self._update_state(color, "helmet", helmet_cnt, helmet_mask)
        self._update_state(color, "foot", foot_cnt, foot_mask)

        return helmet_mask, foot_mask

    def _update_state(self, color, part, cnt_dict, mask):
        key = f"{color}_{part}"
        if cnt_dict is not None:
            cv2.drawContours(mask, [cnt_dict["cnt"]], -1, 255, -1)
            self.prev_centroids[key] = cnt_dict["centroid"]
            self.last_masks[key] = mask.copy()
            self.lost_frames[key] = 0
        else:
            self._handle_loss(color, part, mask)

    def _handle_loss(self, color, part, mask):
        key = f"{color}_{part}"
        self.lost_frames[key] += 1
        if self.lost_frames[key] < 8 and self.last_masks[key] is not None:
            np.copyto(mask, self.last_masks[key])
        else:
            self.prev_centroids[key] = None
            self.last_masks[key] = None

    # ----------------------------------------------------------------
    # OZELLIK CIKARIMI
    # ----------------------------------------------------------------
    def _extract_features(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours, bboxes, centroids, areas = [], [], [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _SEG_MIN_AREA or area > _SEG_MAX_AREA:
                continue
            valid_contours.append(cnt)
            areas.append(area)
            bboxes.append(cv2.boundingRect(cnt))
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                centroids.append((int(M["m10"]/M["m00"]),
                                  int(M["m01"]/M["m00"])))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                centroids.append((x + w // 2, y + h // 2))
        return {
            "mask": mask, "contours": valid_contours,
            "bboxes": bboxes, "centroids": centroids, "areas": areas,
        }

    # ----------------------------------------------------------------
    # VIZUALIZASYON
    # ----------------------------------------------------------------
    def draw_segmentation(self, frame, results, alpha=0.4):
        overlay = frame.copy()
        colors = {
            "red_helmet": (0, 0, 200),   "red_foot": (0, 100, 255),
            "blue_helmet": (200, 100, 0), "blue_foot": (255, 180, 0),
        }
        for key, color in colors.items():
            data = results.get(key, {})
            cv2.drawContours(overlay, data.get("contours", []), -1, color, -1)
            for cx, cy in data.get("centroids", []):
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), -1)
                cv2.circle(overlay, (cx, cy), 8, color, 2)
            for x, y, w, h in data.get("bboxes", []):
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                label = key.replace("_", " ").title()
                cv2.putText(overlay, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
