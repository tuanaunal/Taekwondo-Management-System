"""Temas ölçümü, temas anı ve karar motoru testleri."""
import pytest

from src.contact_analyzer import ContactAnalyzer
from src.contact_moment import frames_to_ranges
from src.decision_engine import DecisionEngine, calibrated_confidence

R = 26.0  # referans baş yarıçapı


def helmet(center=(500, 300), radius=R):
    return {"head_center": center, "head_radius": radius}


@pytest.mark.parametrize("dist_radii, beklenen", [
    (1.2, "overlap"),       # 1.5 yarıçap içi -> temas
    (2.0, "near_miss"),     # 2.5 yarıçap içi -> yakın geçiş
    (4.0, "no_contact"),
])
def test_temas_bas_yaricapi_cinsinden(dist_radii, beklenen):
    """Adım 12: temas kararı ayak ile baş merkezi arasındaki mesafeye (baş yarıçapı) göre verilir."""
    foot = {"foot_tips": [(500 + dist_radii * R, 300)], "centroids": []}
    r = ContactAnalyzer().analyze_contact(helmet(), foot)
    assert r["contact_type"] == beklenen
    assert r["has_contact"] == (beklenen == "overlap")
    assert abs(r["distance_radii"] - dist_radii) < 1e-6


def test_bas_veya_ayak_yoksa_tespit_yok():
    r = ContactAnalyzer().analyze_contact({"head_center": None}, {"foot_tips": [(1, 1)]})
    assert r["contact_type"] == "no_detection"


def test_kare_araliklari():
    """Adım 22: işaretli kareler ardışık aralıklara dönüşür."""
    assert frames_to_ranges([12, 10, 11, 20, 13, 20]) == [(10, 13), (20, 20)]
    assert frames_to_ranges([]) == []


MODEL = {
    "features": ["rotation", "absolute_swing", "contact_gap_radii"],
    "mean": [1.0, 200.0, 10.0], "std": [0.5, 100.0, 4.0],
    "weights": [1.0, 0.5, -0.2], "bias": 0.0, "threshold": 0.5,
    "kalibrasyon": [
        {"alt": 0.0, "ust": 0.10, "karar_sayisi": 10, "dogruluk": 0.53},
        {"alt": 0.10, "ust": 0.25, "karar_sayisi": 10, "dogruluk": 0.56},
        {"alt": 0.25, "ust": 0.51, "karar_sayisi": 10, "dogruluk": 0.80},
    ],
}
STRONG = {"max": {"rotation": 2.5, "absolute_swing": 400}, "coverage": 1.0}   # güçlü baş tepkisi
WEAK = {"max": {"rotation": 0.2, "absolute_swing": 30}, "coverage": 1.0}      # tepki yok
CONTACT = {"min_distance": 10.0}


@pytest.mark.parametrize("head, sensor, beklenen", [
    (STRONG, True, "PUAN_GECERLI"),
    (WEAK, True, "GHOST_HIT"),
    (STRONG, False, "KACIRILAN_VURUS"),
    (WEAK, False, "DOGRU_KARAR"),
    (STRONG, None, "TEMAS_VAR"),
    (WEAK, None, "TEMAS_YOK"),
])
def test_sensor_tablosu(head, sensor, beklenen):
    """Adım 17: görüntü kararı sensör durumuyla 1. adımdaki tabloya göre birleşir."""
    d = DecisionEngine(MODEL).make_decision(CONTACT, {}, head_response=head, sensor_scored=sensor)
    assert d["decision"] == beklenen


def test_goruntu_yetersizse_belirsiz():
    head = dict(STRONG, coverage=0.2)
    d = DecisionEngine(MODEL).make_decision(CONTACT, {}, head_response=head, sensor_scored=True)
    assert d["decision"] == "BELIRSIZ"
    assert d["confidence"] == 0.0


def test_model_yoksa_belirsiz():
    engine = DecisionEngine(MODEL)
    engine.model = None
    d = engine.make_decision(CONTACT, {}, head_response=STRONG)
    assert d["decision"] == "BELIRSIZ"


def test_kalibre_guven_modelin_iddiasi_degil():
    """Adım 19: gösterilen güven, benzer kararların ölçülmüş doğruluğudur."""
    assert calibrated_confidence(0.97, MODEL) == 0.80
    assert calibrated_confidence(0.55, MODEL) == 0.53
    d = DecisionEngine(MODEL).make_decision(CONTACT, {}, head_response=STRONG)
    assert d["confidence"] in (0.53, 0.56, 0.80)
    assert d["confidence_level"] in ("düşük", "orta", "yüksek")


def test_kare_kare_bas_donusu_karardaki_degerle_ayni():
    """Arayüz 2: grafikte çizilen kare kare baş dönüşünün en büyüğü, kararda kullanılan değere eşit olmalı."""
    from src.head_response import compute_head_response

    feats = []
    for i in range(40):
        dx = 0.0 if i < 20 else 15.0          # 20. kareden sonra burun yana kayıyor (baş dönüyor)
        feats.append({"red_helmet": {"centroids": [(500 + dx, 300)], "head_center": (500, 300),
                                     "head_radius": 26.0, "torso_center": (500, 400)},
                      "blue_helmet": {}})
    hr = compute_head_response(feats, 30.0)
    series = [v for v in hr["red"]["rotation_series"] if v is not None]
    assert len(hr["red"]["rotation_series"]) == 40
    assert max(series) == pytest.approx(hr["red"]["rotation"], abs=1e-3)
    assert hr["red"]["rotation"] > 0.3
