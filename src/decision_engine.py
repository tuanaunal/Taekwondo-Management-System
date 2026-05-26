"""
decision_engine.py — Ghost Hit / Real Hit Karar Motoru
=======================================================
Kinematik analiz ve temas analizi sonuçlarını birleştirerek
nihai kararı verir.

Karar Matrisi:
┌──────────────┬───────────────┬──────────────────────────────┐
│ Temas Durumu │ İvme Profili  │ Karar                        │
├──────────────┼───────────────┼──────────────────────────────┤
│ Temas VAR    │ Darbe İvmesi  │ ✅ Gerçek Darbe (Real Hit)   │
│ Temas VAR    │ Aktif Kaçış   │ ⚠️ Hafif Temas               │
│ Temas YOK    │ Darbe İvmesi  │ 🔶 Dış Etken / Yanlış Alarm │
│ Temas YOK    │ Aktif Kaçış   │ 🔴 Ghost Hit                 │
└──────────────┴───────────────┴──────────────────────────────┘
"""

from src.config import WEIGHT_CONTACT, WEIGHT_KINEMATICS, CONFIDENCE_THRESHOLD


class DecisionEngine:
    """
    Temas ve kinematik analiz sonuçlarını birleştirerek
    Ghost Hit / Real Hit kararı veren motor.
    """

    # Karar sabitleri
    REAL_HIT = "REAL_HIT"
    LIGHT_CONTACT = "LIGHT_CONTACT"
    EXTERNAL_FACTOR = "EXTERNAL_FACTOR"
    GHOST_HIT = "GHOST_HIT"
    INCONCLUSIVE = "INCONCLUSIVE"

    # Türkçe etiketler
    LABELS_TR = {
        REAL_HIT: "Gerçek Darbe (Real Hit)",
        LIGHT_CONTACT: "Hafif Temas (Light Contact)",
        EXTERNAL_FACTOR: "Dış Etken / Aktif Kaçış",
        GHOST_HIT: "Ghost Hit (Temassız Sarsıntı)",
        INCONCLUSIVE: "Belirsiz (Yetersiz Veri)",
    }

    # Renk kodları (GUI için)
    COLORS = {
        REAL_HIT: "#2ECC71",         # Yeşil
        LIGHT_CONTACT: "#F39C12",    # Turuncu
        EXTERNAL_FACTOR: "#E67E22",  # Koyu turuncu
        GHOST_HIT: "#E74C3C",        # Kırmızı
        INCONCLUSIVE: "#95A5A6",     # Gri
    }

    # Emoji göstergeleri
    INDICATORS = {
        REAL_HIT: "✅",
        LIGHT_CONTACT: "⚠️",
        EXTERNAL_FACTOR: "🔶",
        GHOST_HIT: "🔴",
        INCONCLUSIVE: "❓",
    }

    def make_decision(
        self,
        contact_summary: dict,
        kinematic_result: dict,
    ) -> dict:
        """
        Nihai kararı verir.

        Parameters
        ----------
        contact_summary : dict
            contact_analyzer.analyze_video_contacts() çıktısı.
        kinematic_result : dict
            kinematics.classify_motion() çıktısı.

        Returns
        -------
        dict
            {
                'decision': str,           # REAL_HIT | GHOST_HIT | ...
                'label_tr': str,           # Türkçe etiket
                'indicator': str,          # Emoji göstergesi
                'color': str,              # Hex renk kodu
                'confidence': float,       # 0.0 - 1.0
                'contact_score': float,    # Temas bileşen skoru
                'kinematic_score': float,  # Kinematik bileşen skoru
                'reasoning': str,          # Karar gerekçesi
                'contact_details': dict,   # Temas detayları
                'kinematic_details': dict, # Kinematik detayları
            }
        """
        # ── Bileşen Skorları ──
        has_contact = contact_summary.get("has_any_contact", False)
        contact_ratio = contact_summary.get("contact_ratio", 0.0)
        max_overlap = contact_summary.get("max_overlap", 0)
        min_distance = contact_summary.get("min_distance", float("inf"))
        near_miss_frames = contact_summary.get("near_miss_frames", 0)
        total_frames = contact_summary.get("total_frames", 1)

        # Near miss oranı — overlap olmasa bile yakın geçiş temas kanıtıdır
        near_miss_ratio = near_miss_frames / max(total_frames, 1)
        # Etkili temas oranı: overlap + near miss'in yarısı
        effective_contact_ratio = contact_ratio + near_miss_ratio * 0.5

        motion_type = kinematic_result.get("type", "stationary")
        motion_confidence = kinematic_result.get("confidence", 0.0)
        max_acceleration = kinematic_result.get("max_acceleration", 0.0)

        # Temas skoru (0: temas yok, 1: güçlü temas)
        if has_contact:
            contact_score = min(1.0, 0.4 + effective_contact_ratio * 0.6)
        elif min_distance < 50:
            contact_score = 0.3 * (1.0 - min_distance / 50.0)
        else:
            contact_score = 0.0

        # Kinematik skoru (0: darbe, 1: kaçış)
        if motion_type == "impact":
            kinematic_score = 0.0  # Düşük = darbe var
        elif motion_type == "evasion":
            kinematic_score = 1.0  # Yüksek = kaçış
        else:
            kinematic_score = 0.5  # Belirsiz

        net_disp = kinematic_result.get("net_displacement", 0.0)

        # ── Karar Matrisi ──
        if motion_type == "impact":
            # Makine öğrenmesi analizinden çıkan optimum kural: 
            # 8 piksel altı yakınlaşma GERÇEK temastır (YOLO hata payı).
            if has_contact or min_distance <= 8.0:
                # Gerçek fiziksel temas veya çok yakın teğet geçiş + İvme
                if net_disp > 30.0 or max_acceleration > 2000.0:
                    decision = self.REAL_HIT
                    confidence = 0.85
                    reasoning = f"Fiziksel temas (veya 8px altı yakınlaşma) ve darbe ivmesi doğrulandı. Kesin Gerçek Darbe."
                else:
                    decision = self.LIGHT_CONTACT
                    confidence = 0.70
                    reasoning = f"Temas var ancak darbe ivmesi/savrulma yetersiz. Hafif temas/Sürtünme."
            else:
                # Temas yok. İkisini ayırmak için net_disp (kafa savrulması) kullanıyoruz.
                if net_disp > 150.0:
                    decision = self.REAL_HIT
                    confidence = 0.8
                    reasoning = f"Temas tespit edilemedi ancak kaskta istikrarlı savrulma (Savrulma: {net_disp:.1f} > 150) var. Gerçek darbe."
                else:
                    decision = self.GHOST_HIT
                    confidence = 0.8
                    reasoning = f"Temas yok. Kaskta sarsıntı var ancak kafa savrulması yetersiz veya hatalı (Savrulma: {net_disp:.1f} <= 150). Ghost Hit."

        elif motion_type in ("evasion", "stationary"):
            if has_contact:
                decision = self.LIGHT_CONTACT
                confidence = contact_score * 0.7
                reasoning = f"Fiziksel temas tespit edildi ancak darbe ivmesi yok. Hafif temas."
            else:
                decision = self.EXTERNAL_FACTOR
                confidence = (1.0 - contact_score) * 0.7
                reasoning = f"Fiziksel temas yok ve kask hareketi kaçış profili gösteriyor. Dış Etken/Kaçış."
        else:
            decision = self.INCONCLUSIVE
            confidence = 0.3
            reasoning = "Yeterli veri bulunamadı veya analiz sonuçları belirsiz."

        # Güven sınırlandırması
        confidence = min(1.0, max(0.0, confidence))

        return {
            "decision": decision,
            "label_tr": self.LABELS_TR[decision],
            "indicator": self.INDICATORS[decision],
            "color": self.COLORS[decision],
            "confidence": confidence,
            "contact_score": contact_score,
            "kinematic_score": kinematic_score,
            "reasoning": reasoning,
            "contact_details": contact_summary,
            "kinematic_details": kinematic_result,
        }

    def format_report(self, result: dict) -> str:
        """Karar sonucunu okunabilir metin raporuna dönüştürür."""
        lines = [
            "=" * 60,
            "  TAEKWONDO GHOST HIT DETECTION — KARAR RAPORU",
            "=" * 60,
            "",
            f"  Karar:      {result['indicator']} {result['label_tr']}",
            f"  Güven:      {result['confidence']:.1%}",
            "",
            "─" * 60,
            "  GEREKÇE",
            "─" * 60,
            f"  {result['reasoning']}",
            "",
            "─" * 60,
            "  DETAYLAR",
            "─" * 60,
        ]

        # Temas detayları
        cd = result.get("contact_details", {})
        lines.extend([
            f"  Temas Analizi:",
            f"    Toplam kare:       {cd.get('total_frames', 0)}",
            f"    Temaslı kare:      {cd.get('contact_frames', 0)}",
            f"    Yakın geçiş:       {cd.get('near_miss_frames', 0)}",
            f"    Maks çakışma:      {cd.get('max_overlap', 0)} piksel",
            f"    Min mesafe:        {cd.get('min_distance', 'N/A')}",
            f"    Temas oranı:       {cd.get('contact_ratio', 0):.1%}",
            "",
        ])

        # Kinematik detaylar
        kd = result.get("kinematic_details", {})
        lines.extend([
            f"  Kinematik Analiz:",
            f"    Hareket tipi:      {kd.get('type', 'N/A')}",
            f"    Maks ivme:         {kd.get('max_acceleration', 0):.1f} px/s²",
            f"    Tepe karesi:       {kd.get('peak_frame', 'N/A')}",
            f"    Detay:             {kd.get('details', 'N/A')}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
