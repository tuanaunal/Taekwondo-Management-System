"""
contact_analyzer.py — Ayak-Baş Temas Ölçümü
=============================================
Saldıran sporcunun ayak noktaları (ayak ucu + ayak bileği) ile rakibin
baş bölgesi (baş merkezi + tahmini baş yarıçapı) arasındaki mesafeyi ölçer.

Eskiden kask ve ayağın etrafına çizilen 80 px'lik sahte dairelerin piksel
çakışması kullanılıyordu; bu daireler gerçek baş/ayak boyutunu yansıtmıyordu.

Tüm mesafeler "referans piksel" cinsindendir (26 px = bir baş yarıçapı).
Not: İki boyutlu görüntüde temas, temassız geçişten ayırt edilemeyebilir
(YOL_HARITASI.md, Bulgu 3); bu ölçüm kararda destekleyici kanıttır.
"""

import numpy as np
from src.config import CONTACT_MARGIN_RADII, NEAR_MISS_MARGIN_RADII


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
        Tek bir kare için ayak-baş temas analizi yapar.

        Parameters
        ----------
        helmet_data : dict
            Rakibin kask verisi; "head_center" ve "head_radius" kullanılır.
        foot_data : dict
            Saldıranın ayak verisi; "foot_tips" (ayak ucu) ve "centroids" (ayak bileği) kullanılır.

        Returns
        -------
        dict
            {
                'has_contact': bool,
                'contact_type': 'overlap' | 'near_miss' | 'no_contact' | 'no_detection',
                'overlap_pixels': int,          # ayağın baş bölgesine giriş derinliği (referans px)
                'centroid_distance': float,     # en yakın ayak noktası - baş merkezi (referans px)
                'min_contour_distance': float,  # en yakın ayak noktası - baş kenarı (referans px, >= 0)
                'distance_radii': float,        # en yakın ayak noktası - baş merkezi (baş yarıçapı)
                'confidence': float,
                'details': str,
            }
        Not: 'overlap' adı geriye dönük uyumluluk için korunmuştur ve "temas" anlamındadır.
        """
        result = {
            "has_contact": False,
            "contact_type": "no_detection",
            "overlap_pixels": 0,
            "centroid_distance": float("inf"),
            "min_contour_distance": float("inf"),
            "distance_radii": float("inf"),
            "confidence": 0.0,
            "details": "",
        }

        center = helmet_data.get("head_center")
        radius = helmet_data.get("head_radius")
        points = list(foot_data.get("foot_tips", [])) + list(foot_data.get("centroids", []))
        if center is None or not radius or not points:
            result["details"] = "Baş veya ayak tespit edilemedi."
            return result

        dist = min(float(np.hypot(p[0] - center[0], p[1] - center[1])) for p in points)
        radii = dist / radius
        gap = max(0.0, dist - radius)

        result["centroid_distance"] = dist
        result["min_contour_distance"] = gap
        result["distance_radii"] = radii
        result["overlap_pixels"] = int(round(max(0.0, radius - dist)))

        if radii <= 1.0 + CONTACT_MARGIN_RADII:
            result["has_contact"] = True
            result["contact_type"] = "overlap"
            result["confidence"] = 1.0
            result["details"] = f"Temas: ayak, baş merkezine {radii:.2f} baş yarıçapı uzaklıkta."
        elif radii <= 1.0 + NEAR_MISS_MARGIN_RADII:
            result["contact_type"] = "near_miss"
            span = NEAR_MISS_MARGIN_RADII - CONTACT_MARGIN_RADII
            result["confidence"] = 1.0 - (radii - 1.0 - CONTACT_MARGIN_RADII) / span
            result["details"] = f"Yakın geçiş: ayak, baş merkezine {radii:.2f} baş yarıçapı uzaklıkta."
        else:
            result["contact_type"] = "no_contact"
            result["confidence"] = min(1.0, (radii - 1.0 - NEAR_MISS_MARGIN_RADII) / 2.0)
            result["details"] = f"Temas yok: ayak, baş merkezine {radii:.2f} baş yarıçapı uzaklıkta."

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
