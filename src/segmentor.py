"""
segmentor.py — HSV Renk Tabanlı Nesne Segmentasyonu
=====================================================
Kask ve ayak koruyucuları HSV renk uzayında ayrıştırır,
morfolojik operasyonlarla nesne bütünlüğünü sağlar ve
konturları çıkarır.
"""

import cv2
import numpy as np
from src.config import (
    HSV_RANGES,
    MORPH_KERNEL_SIZE, MORPH_ERODE_ITERATIONS,
    MORPH_DILATE_ITERATIONS, MORPH_CLOSE_ITERATIONS,
    MIN_CONTOUR_AREA, MAX_CONTOUR_AREA,
)


class EquipmentSegmentor:
    """
    Taekwondo ekipmanlarını (kask, ayak koruyucu) renk tabanlı
    segmentasyon ile ayırt eden sınıf.
    """

    def __init__(self):
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE
        )

    # ────────────────────────────────────
    # ANA SEGMENTASYON
    # ────────────────────────────────────
    def segment_frame(self, frame: np.ndarray) -> dict:
        """
        Verilen BGR karedeki tüm ekipmanları segmente eder.

        Parameters
        ----------
        frame : np.ndarray
            Ön-işlenmiş BGR kare.

        Returns
        -------
        dict
            Her ekipman tipi için:
              - 'mask': İkili maske (0/255)
              - 'contours': Kontur listesi
              - 'bboxes': Sınır kutuları [(x,y,w,h), ...]
              - 'centroids': Merkez noktaları [(cx,cy), ...]
              - 'areas': Kontur alanları [int, ...]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = {}

        # ── Kırmızı Ekipman (Kask + Ayak) ──
        red_mask = self._create_red_mask(hsv)
        red_mask = self._apply_morphology(red_mask)
        red_helmet, red_foot = self._separate_helmet_foot(red_mask, frame)

        results["red_helmet"] = self._extract_features(red_helmet)
        results["red_foot"] = self._extract_features(red_foot)

        # ── Mavi Ekipman (Kask + Ayak) ──
        blue_mask = self._create_blue_mask(hsv)
        blue_mask = self._apply_morphology(blue_mask)
        blue_helmet, blue_foot = self._separate_helmet_foot(blue_mask, frame)

        results["blue_helmet"] = self._extract_features(blue_helmet)
        results["blue_foot"] = self._extract_features(blue_foot)

        return results

    # ────────────────────────────────────
    # RENK MASKELERİ
    # ────────────────────────────────────
    def _create_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Kırmızı renk için iki ayrı HSV aralığını birleştirir.
        Kırmızı, HSV'de 0° ve 180° civarında iki parçadır.
        """
        mask1 = cv2.inRange(
            hsv,
            HSV_RANGES["red_helmet_lower1"],
            HSV_RANGES["red_helmet_upper1"],
        )
        mask2 = cv2.inRange(
            hsv,
            HSV_RANGES["red_helmet_lower2"],
            HSV_RANGES["red_helmet_upper2"],
        )
        return cv2.bitwise_or(mask1, mask2)

    def _create_blue_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Mavi renk için HSV maskesi oluşturur."""
        return cv2.inRange(
            hsv,
            HSV_RANGES["blue_helmet_lower"],
            HSV_RANGES["blue_helmet_upper"],
        )

    # ────────────────────────────────────
    # MORFOLOJİK İŞLEMLER
    # ────────────────────────────────────
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Erosion → Dilation → Close sırasıyla morfolojik operasyonlar.
        - Erosion: Küçük gürültü parçacıklarını siler
        - Dilation: Nesne alanını genişletir, boşlukları doldurur
        - Close: Kalan küçük delikleri kapatır
        """
        mask = cv2.erode(
            mask, self.morph_kernel,
            iterations=MORPH_ERODE_ITERATIONS,
        )
        mask = cv2.dilate(
            mask, self.morph_kernel,
            iterations=MORPH_DILATE_ITERATIONS,
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self.morph_kernel,
            iterations=MORPH_CLOSE_ITERATIONS,
        )
        return mask

    # ────────────────────────────────────
    # KASK – AYAK AYIRIMI
    # ────────────────────────────────────
    def _separate_helmet_foot(
        self, mask: np.ndarray, frame: np.ndarray
    ) -> tuple:
        """
        Aynı renkteki kask ve ayak koruyucuyu konum ve boyut bazlı ayırır.

        Mantık:
          - Kask genellikle karenin üst yarısında
          - Ayak koruyucu genellikle karenin alt yarısında
          - MIN_CONTOUR_AREA ve MAX_CONTOUR_AREA filtresi uygulanır
          - Aspect ratio kontrolü: kask yuvarlak, ayak uzun

        Returns
        -------
        tuple
            (helmet_mask, foot_mask) — iki ayrı ikili maske.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = mask.shape[:2]
        mid_y = h // 2

        helmet_mask = np.zeros_like(mask)
        foot_mask = np.zeros_like(mask)

        upper_contours = []
        lower_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Boyut filtreleme: çok küçük veya çok büyük konturları reddet
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            # Aspect ratio kontrolü
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect_ratio = bw / bh

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cy = int(M["m01"] / M["m00"])

            if cy < mid_y:
                # Üst yarı: Kask adayı
                # Kask genellikle yuvarlak/oval: aspect ratio 0.4 - 2.5
                if 0.3 <= aspect_ratio <= 3.0:
                    upper_contours.append((cnt, area))
            else:
                # Alt yarı: Ayak koruyucu adayı
                lower_contours.append((cnt, area))

        # Üst yarıdaki en büyük kontur → kask
        if upper_contours:
            upper_contours.sort(key=lambda x: x[1], reverse=True)
            cv2.drawContours(helmet_mask, [upper_contours[0][0]], -1, 255, -1)

        # Alt yarıdaki en büyük kontur → ayak koruyucu
        if lower_contours:
            lower_contours.sort(key=lambda x: x[1], reverse=True)
            cv2.drawContours(foot_mask, [lower_contours[0][0]], -1, 255, -1)

        # Eğer sadece bir yarıda nesne varsa, konumu daha hassas kontrol et
        if not upper_contours and lower_contours and len(lower_contours) >= 2:
            # İkinci en büyük kontur kask olabilir
            cv2.drawContours(helmet_mask, [lower_contours[1][0]], -1, 255, -1)
        elif not lower_contours and upper_contours and len(upper_contours) >= 2:
            cv2.drawContours(foot_mask, [upper_contours[1][0]], -1, 255, -1)

        return helmet_mask, foot_mask

    # ────────────────────────────────────
    # ÖZELLİK ÇIKARIMI
    # ────────────────────────────────────
    def _extract_features(self, mask: np.ndarray) -> dict:
        """
        Verilen maskeden konturları, merkez noktalarını,
        sınır kutularını ve alanları çıkarır.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        bboxes = []
        centroids = []
        areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            valid_contours.append(cnt)
            areas.append(area)
            bboxes.append(cv2.boundingRect(cnt))

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                centroids.append((x + w // 2, y + h // 2))

        return {
            "mask": mask,
            "contours": valid_contours,
            "bboxes": bboxes,
            "centroids": centroids,
            "areas": areas,
        }

    # ────────────────────────────────────
    # VİZÜALİZASYON
    # ────────────────────────────────────
    def draw_segmentation(
        self, frame: np.ndarray, results: dict, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Segmentasyon sonuçlarını orijinal kare üzerine yarı-saydam çizer.
        """
        overlay = frame.copy()
        colors = {
            "red_helmet": (0, 0, 200),
            "red_foot": (0, 100, 255),
            "blue_helmet": (200, 100, 0),
            "blue_foot": (255, 180, 0),
        }

        for key, color in colors.items():
            data = results.get(key, {})
            contours = data.get("contours", [])
            centroids = data.get("centroids", [])
            bboxes = data.get("bboxes", [])

            # Konturları doldur
            cv2.drawContours(overlay, contours, -1, color, -1)

            # Merkez noktalarını çiz
            for cx, cy in centroids:
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), -1)
                cv2.circle(overlay, (cx, cy), 8, color, 2)

            # Sınır kutularını çiz
            for x, y, w, h in bboxes:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

            # Etiket yaz
            for (x, y, w, h) in bboxes:
                label = key.replace("_", " ").title()
                cv2.putText(
                    overlay, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        # Yarı-saydam birleşim
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return result
