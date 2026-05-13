"""
contact_analyzer.py — Piksel Bazlı Temas Doğrulama
====================================================
Kask ve ayak koruyucu maskeleri arasındaki fiziksel teması
piksel çakışması, Öklid mesafesi ve kontur sınır mesafesi ile ölçer.
"""

import cv2
import numpy as np
from src.config import (
    CONTACT_OVERLAP_THRESHOLD,
    PROXIMITY_THRESHOLD,
    CONTOUR_DISTANCE_THRESHOLD,
)


class ContactAnalyzer:
    """
    İki nesne (kask ve ayak koruyucu) arasındaki fiziksel
    temas durumunu analiz eder.
    """

    def analyze_contact(
        self,
        helmet_data: dict,
        foot_data: dict,
    ) -> dict:
        """
        Tek bir kare için temas analizi yapar.

        Parameters
        ----------
        helmet_data : dict
            Kask segmentasyon verisi (mask, contours, centroids).
        foot_data : dict
            Ayak koruyucu segmentasyon verisi.

        Returns
        -------
        dict
            {
                'has_contact': bool,
                'contact_type': 'overlap' | 'near_miss' | 'no_contact' | 'no_detection',
                'overlap_pixels': int,
                'centroid_distance': float,
                'min_contour_distance': float,
                'confidence': float,
                'details': str,
            }
        """
        result = {
            "has_contact": False,
            "contact_type": "no_detection",
            "overlap_pixels": 0,
            "centroid_distance": float("inf"),
            "min_contour_distance": float("inf"),
            "confidence": 0.0,
            "details": "",
        }

        # Nesneler tespit edilmemişse
        helmet_mask = helmet_data.get("mask")
        foot_mask = foot_data.get("mask")
        helmet_centroids = helmet_data.get("centroids", [])
        foot_centroids = foot_data.get("centroids", [])
        helmet_contours = helmet_data.get("contours", [])
        foot_contours = foot_data.get("contours", [])

        if helmet_mask is None or foot_mask is None:
            result["details"] = "Maske verisi eksik."
            return result

        if len(helmet_centroids) == 0 or len(foot_centroids) == 0:
            result["details"] = "Bir veya her iki nesne tespit edilemedi."
            return result

        # ── 1. Piksel Çakışma Testi (Overlap) ──
        overlap = cv2.bitwise_and(helmet_mask, foot_mask)
        overlap_pixels = cv2.countNonZero(overlap)
        result["overlap_pixels"] = overlap_pixels

        # ── 2. Merkez Noktaları Arası Öklid Mesafesi ──
        hc = np.array(helmet_centroids[0], dtype=float)
        fc = np.array(foot_centroids[0], dtype=float)
        centroid_dist = np.linalg.norm(hc - fc)
        result["centroid_distance"] = float(centroid_dist)

        # ── 3. Kontur Sınır Mesafesi ──
        min_contour_dist = self._compute_min_contour_distance(
            helmet_contours, foot_contours
        )
        result["min_contour_distance"] = min_contour_dist

        # ── Karar ──
        if overlap_pixels >= CONTACT_OVERLAP_THRESHOLD:
            result["has_contact"] = True
            result["contact_type"] = "overlap"
            result["confidence"] = min(1.0, overlap_pixels / 50.0)
            result["details"] = (
                f"Fiziksel temas tespit edildi! "
                f"Çakışan piksel: {overlap_pixels}, "
                f"Merkez mesafesi: {centroid_dist:.1f}px"
            )
        elif min_contour_dist < CONTOUR_DISTANCE_THRESHOLD:
            result["has_contact"] = False
            result["contact_type"] = "near_miss"
            result["confidence"] = 1.0 - (min_contour_dist / CONTOUR_DISTANCE_THRESHOLD)
            result["details"] = (
                f"Yakın geçiş (Near Miss). "
                f"Kontur mesafesi: {min_contour_dist:.1f}px, "
                f"Merkez mesafesi: {centroid_dist:.1f}px"
            )
        else:
            result["has_contact"] = False
            result["contact_type"] = "no_contact"
            result["confidence"] = min(1.0, min_contour_dist / PROXIMITY_THRESHOLD)
            result["details"] = (
                f"Temas yok. "
                f"Kontur mesafesi: {min_contour_dist:.1f}px, "
                f"Merkez mesafesi: {centroid_dist:.1f}px"
            )

        return result

    def analyze_video_contacts(
        self,
        frame_results: list,
    ) -> dict:
        """
        Tüm kareler üzerindeki temas analizlerini özetler.

        Parameters
        ----------
        frame_results : list of dict
            Her kare için analyze_contact() çıktıları.

        Returns
        -------
        dict
            {
                'total_frames': int,
                'contact_frames': int,
                'near_miss_frames': int,
                'no_contact_frames': int,
                'max_overlap': int,
                'min_distance': float,
                'contact_ratio': float,
                'has_any_contact': bool,
                'peak_contact_frame': int,
                'distance_profile': list,
                'overlap_profile': list,
            }
        """
        total = len(frame_results)
        if total == 0:
            return {
                "total_frames": 0,
                "contact_frames": 0,
                "near_miss_frames": 0,
                "no_contact_frames": 0,
                "max_overlap": 0,
                "min_distance": float("inf"),
                "contact_ratio": 0.0,
                "has_any_contact": False,
                "peak_contact_frame": -1,
                "distance_profile": [],
                "overlap_profile": [],
            }

        contact_count = sum(
            1 for r in frame_results if r["contact_type"] == "overlap"
        )
        near_miss_count = sum(
            1 for r in frame_results if r["contact_type"] == "near_miss"
        )
        no_contact_count = sum(
            1 for r in frame_results if r["contact_type"] == "no_contact"
        )

        overlaps = [r["overlap_pixels"] for r in frame_results]
        distances = [r["min_contour_distance"] for r in frame_results]

        max_overlap = max(overlaps) if overlaps else 0
        min_distance = min(d for d in distances if d < float("inf")) if any(
            d < float("inf") for d in distances
        ) else float("inf")

        peak_frame = overlaps.index(max_overlap) if max_overlap > 0 else -1

        return {
            "total_frames": total,
            "contact_frames": contact_count,
            "near_miss_frames": near_miss_count,
            "no_contact_frames": no_contact_count,
            "max_overlap": max_overlap,
            "min_distance": min_distance,
            "contact_ratio": contact_count / total if total > 0 else 0.0,
            "has_any_contact": contact_count > 0,
            "peak_contact_frame": peak_frame,
            "distance_profile": distances,
            "overlap_profile": overlaps,
        }

    # ────────────────────────────────────
    # YARDIMCI: Kontur Mesafesi Hesabı
    # ────────────────────────────────────
    def _compute_min_contour_distance(
        self,
        contours_a: list,
        contours_b: list,
    ) -> float:
        """
        İki kontur grubu arasındaki minimum mesafeyi hesaplar.

        Her iki kontur grubunun sınır noktaları arasındaki
        en yakın nokta çiftinin Öklid mesafesini döner.
        """
        if not contours_a or not contours_b:
            return float("inf")

        min_dist = float("inf")

        for cnt_a in contours_a:
            for cnt_b in contours_b:
                # Kontur noktalarını düzleştir
                pts_a = cnt_a.reshape(-1, 2).astype(float)
                pts_b = cnt_b.reshape(-1, 2).astype(float)

                # Her A noktası için en yakın B noktasını bul
                # Performans optimizasyonu: alt örnekleme
                step_a = max(1, len(pts_a) // 50)
                step_b = max(1, len(pts_b) // 50)
                sampled_a = pts_a[::step_a]
                sampled_b = pts_b[::step_b]

                for pa in sampled_a:
                    dists = np.linalg.norm(sampled_b - pa, axis=1)
                    d = np.min(dists)
                    if d < min_dist:
                        min_dist = d

        return min_dist
