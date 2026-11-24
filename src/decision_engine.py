"""
decision_engine.py — Karar Motoru
==================================

PROBLEM TANIMI
--------------
Kafa vuruşunun puanı, ayak üstü koruyucusundaki sensör ile kasktaki
sensörün eşleşmesiyle elektronik olarak verilir. İki tür hata oluşur:
  - Ghost Hit:       Sensör puan verdi, ama kafaya gerçek temas yok.
  - Kaçırılan vuruş: Kafaya gerçek temas var, ama sensör puan vermedi.

Sistemin sorusu (sensörden bağımsız, sadece görüntüye bakarak):
  "Tartışmalı anda kafaya gerçek bir temas oldu mu?"
  -> TEMAS VAR / TEMAS YOK / BELİRSİZ

KARAR YAPISI
------------
1. Görüntü kararı (sensörden bağımsız):
   Başın darbeye verdiği tepki ölçülür (baş dönüşü, burun savrulması) ve
   destekleyici kanıt olarak ayak-baş yakınlığı eklenir. Bu ölçümler, veriden
   öğrenilmiş küçük ve yorumlanabilir bir modelle (lojistik regresyon)
   "temas var" olasılığına çevrilir. Model dosyası: model/karar_modeli.json
   (eğitim: egit.py). Görüntü karar vermeye yetmiyorsa sonuç BELİRSİZ'dir.

2. Sensör bilgisiyle birleştirme (sensör durumu verilmişse):
┌─────────────────────┬──────────────────────┬───────────────────────────┐
│                     │ Görüntü: TEMAS VAR   │ Görüntü: TEMAS YOK        │
├─────────────────────┼──────────────────────┼───────────────────────────┤
│ Sensör puan verdi   │ ✅ Puan geçerli      │ 🔴 Ghost Hit (puan hatalı)│
│ Sensör puan vermedi │ 🟠 Kaçırılan vuruş   │ ✅ Puan yok, doğru karar  │
└─────────────────────┴──────────────────────┴───────────────────────────┘
   Sensör durumu verilmemişse sonuç görüntü kararıdır.
"""

import json
import os

import numpy as np

from src.config import PROJECT_ROOT

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "karar_modeli.json")

# Görüntünün karar vermeye yetmesi için: hedef sporcunun başı kliplerin en az bu oranında
# gerçekten ölçülmüş olmalı (ölçülmeyip son konumu tutulan kareler sayılmaz).
MIN_HEAD_COVERAGE = 0.5

# Kararda kullanılan ölçümler (sıra modeldekiyle aynı olmalı) ve Türkçe adları
FEATURES = ["rotation", "absolute_swing", "contact_gap_radii"]
FEATURE_LABELS_TR = {
    "rotation": "baş dönüşü",
    "absolute_swing": "burun savrulması",
    "contact_gap_radii": "ayak-baş mesafesi",
}
FEATURE_UNITS_TR = {
    "rotation": "baş yarıçapı",
    "absolute_swing": "referans px",
    "contact_gap_radii": "baş yarıçapı",
}
CONTACT_GAP_CAP = 20.0  # Ayak hiç görülemezse mesafe bu değerle sınırlanır (baş yarıçapı)
REFERENCE_HEAD_RADIUS = 26.0


def extract_features(contact_summary: dict, head_response: dict) -> dict:
    """Analiz sonucundan karar modelinin kullandığı ölçümleri çıkarır."""
    hr = head_response.get("max", {}) if head_response else {}
    gap_px = contact_summary.get("min_distance", float("inf")) if contact_summary else float("inf")
    gap_radii = gap_px / REFERENCE_HEAD_RADIUS if np.isfinite(gap_px) else CONTACT_GAP_CAP
    return {
        "rotation": float(hr.get("rotation", 0.0)),
        "absolute_swing": float(hr.get("absolute_swing", 0.0)),
        "contact_gap_radii": float(min(gap_radii, CONTACT_GAP_CAP)),
    }


def load_model(path: str = MODEL_PATH):
    """Eğitilmiş karar modelini yükler; dosya yoksa None döner."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict_contact_probability(features: dict, model: dict) -> tuple:
    """
    Lojistik regresyon: p(temas var) = 1 / (1 + e^-(b + Σ w_i * z_i)),
    z_i = (ölçüm_i - ortalama_i) / std_i.
    Dönüş: (olasılık, her ölçümün katkısı)
    """
    z = {}
    for name, mean, std in zip(model["features"], model["mean"], model["std"]):
        z[name] = (features[name] - mean) / (std if std > 0 else 1.0)
    contributions = {name: w * z[name] for name, w in zip(model["features"], model["weights"])}
    logit = model["bias"] + sum(contributions.values())
    return 1.0 / (1.0 + np.exp(-logit)), contributions


class DecisionEngine:
    """Görüntü ölçümlerinden temas kararı veren ve sensör bilgisiyle birleştiren motor."""

    # Görüntü kararları
    CONTACT = "TEMAS_VAR"
    NO_CONTACT = "TEMAS_YOK"
    UNCERTAIN = "BELIRSIZ"
    # Sensör bilgisiyle birleşmiş kararlar
    VALID_SCORE = "PUAN_GECERLI"
    GHOST_HIT = "GHOST_HIT"
    MISSED_HIT = "KACIRILAN_VURUS"
    CORRECT_NO_SCORE = "DOGRU_KARAR"

    LABELS_TR = {
        CONTACT: "Temas Var (Gerçek Darbe)",
        NO_CONTACT: "Temas Yok",
        UNCERTAIN: "Belirsiz (Görüntü Yetersiz)",
        VALID_SCORE: "Puan Geçerli",
        GHOST_HIT: "Ghost Hit (Puan Hatalı)",
        MISSED_HIT: "Kaçırılan Vuruş (Puan Verilmeli)",
        CORRECT_NO_SCORE: "Puan Yok – Doğru Karar",
    }
    COLORS = {
        CONTACT: "#2ECC71", NO_CONTACT: "#E74C3C", UNCERTAIN: "#95A5A6",
        VALID_SCORE: "#2ECC71", GHOST_HIT: "#E74C3C",
        MISSED_HIT: "#F39C12", CORRECT_NO_SCORE: "#2ECC71",
    }
    INDICATORS = {
        CONTACT: "✅", NO_CONTACT: "🔴", UNCERTAIN: "❓",
        VALID_SCORE: "✅", GHOST_HIT: "🔴", MISSED_HIT: "🟠", CORRECT_NO_SCORE: "✅",
    }

    def __init__(self, model: dict = None):
        self.model = model if model is not None else load_model()

    def make_decision(self, contact_summary: dict, kinematic_result: dict,
                      head_response: dict = None, sensor_scored: bool = None) -> dict:
        """
        Nihai kararı verir.

        Parameters
        ----------
        contact_summary : contact_analyzer.analyze_video_contacts() çıktısı
        kinematic_result : kinematics.classify_motion() çıktısı (raporlama için)
        head_response : head_response.compute_head_response() çıktısı
        sensor_scored : True = sensör puan verdi, False = vermedi, None = bilinmiyor

        Returns
        -------
        dict
            decision, label_tr, indicator, color: nihai karar (sensör varsa birleşik)
            contact_decision: görüntü kararı (TEMAS_VAR / TEMAS_YOK / BELIRSIZ)
            contact_probability: modelin "temas var" olasılığı (0-1) veya None
            confidence: kararın güveni (0-1)
            reasoning: Türkçe gerekçe
            features, contributions: kullanılan ölçümler ve karara katkıları
        """
        head_response = head_response or {}
        features = extract_features(contact_summary, head_response)
        coverage = head_response.get("coverage", 1.0)

        probability, contributions = None, {}
        if self.model is None:
            contact_decision = self.UNCERTAIN
            confidence = 0.0
            reasoning = "Karar modeli bulunamadı (model/karar_modeli.json). Önce 'python egit.py' çalıştırılmalı."
        elif coverage < MIN_HEAD_COVERAGE:
            contact_decision = self.UNCERTAIN
            confidence = 0.0
            reasoning = (f"Görüntü yetersiz: hedef sporcunun başı karelerin yalnızca %{coverage * 100:.0f}'inde "
                         f"ölçülebildi (en az %{MIN_HEAD_COVERAGE * 100:.0f} gerekli).")
        else:
            probability, contributions = predict_contact_probability(features, self.model)
            threshold = self.model.get("threshold", 0.5)
            contact_decision = self.CONTACT if probability >= threshold else self.NO_CONTACT
            confidence = probability if contact_decision == self.CONTACT else 1.0 - probability
            reasoning = self._reasoning(contact_decision, probability, features, contributions)

        decision = self._combine_with_sensor(contact_decision, sensor_scored)
        if sensor_scored is not None and contact_decision != self.UNCERTAIN:
            sensor_txt = "Sensör puan verdi" if sensor_scored else "Sensör puan vermedi"
            reasoning = f"{sensor_txt}; görüntüye göre {self.LABELS_TR[contact_decision].lower()}. " + reasoning

        return {
            "decision": decision,
            "label_tr": self.LABELS_TR[decision],
            "indicator": self.INDICATORS[decision],
            "color": self.COLORS[decision],
            "contact_decision": contact_decision,
            "contact_probability": probability,
            "confidence": float(confidence),
            "sensor_scored": sensor_scored,
            "reasoning": reasoning,
            "features": features,
            "contributions": contributions,
            "contact_details": contact_summary,
            "kinematic_details": kinematic_result,
        }

    def _combine_with_sensor(self, contact_decision: str, sensor_scored) -> str:
        if sensor_scored is None or contact_decision == self.UNCERTAIN:
            return contact_decision
        if sensor_scored:
            return self.VALID_SCORE if contact_decision == self.CONTACT else self.GHOST_HIT
        return self.MISSED_HIT if contact_decision == self.CONTACT else self.CORRECT_NO_SCORE

    def _reasoning(self, decision, probability, features, contributions) -> str:
        """Kararı en çok etkileyen ölçümlerden Türkçe gerekçe üretir."""
        verdict = "temas olduğunu" if decision == self.CONTACT else "temas olmadığını"
        ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        parts = []
        for name, c in ordered:
            direction = "temas lehine" if c > 0 else "temassızlık lehine"
            parts.append(f"{FEATURE_LABELS_TR[name]} {features[name]:.2f} {FEATURE_UNITS_TR[name]} ({direction})")
        return (f"Model %{probability * 100:.0f} olasılıkla {verdict} değerlendiriyor. "
                f"Etkisi en büyükten küçüğe: " + "; ".join(parts) + ".")

    def format_report(self, result: dict) -> str:
        """Karar sonucunu okunabilir metin raporuna dönüştürür."""
        lines = [
            "=" * 60,
            "  TAEKWONDO HAKEM KARAR DESTEK — KARAR RAPORU",
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
            "  ÖLÇÜMLER",
            "─" * 60,
        ]
        for name, value in result.get("features", {}).items():
            lines.append(f"    {FEATURE_LABELS_TR[name]:20s} {value:8.2f} {FEATURE_UNITS_TR[name]}")
        lines.append("=" * 60)
        return "\n".join(lines)
