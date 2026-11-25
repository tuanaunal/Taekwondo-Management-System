"""Kinematik analiz testleri (YOLO / video gerekmez)."""
import numpy as np

from src.kinematics import KinematicAnalyzer


def trajectory(xy):
    return [(float(x), float(y), i) for i, (x, y) in enumerate(xy)]


def test_sert_darbede_takip_donmaz():
    """Adım A2: kafa kalıcı olarak 60 px savrulunca konum eskisi gibi 500'de donmamalı."""
    xy = [(500, 300)] * 10 + [(560 + 2 * i, 300) for i in range(20)]
    pos = KinematicAnalyzer(30).kinematic_series(trajectory(xy))["position"]
    assert pos[-1][0] > 590


def test_tek_karelik_hata_temizlenir():
    """Bir karede 200 px sıçrayıp geri dönen tekil hata düzeltilmeli."""
    xy = [(500, 300)] * 30
    xy[15] = (700, 300)
    raw = KinematicAnalyzer(30).kinematic_series(trajectory(xy))["raw"]
    assert abs(raw[15][0] - 500) < 1e-6


def test_ivme_fps_den_bagimsiz():
    """Adım 5: aynı fiziksel hareket 30 ve 60 FPS'te aynı ivmeyi vermeli (600 px/s²)."""
    for fps in (30.0, 60.0):
        t = np.arange(int(1.3 * fps)) / fps
        xy = np.column_stack([500 + 300 * t ** 2, np.full_like(t, 300.0)])
        acc = KinematicAnalyzer(fps).compute_acceleration(trajectory(xy))
        assert abs(np.median(acc) - 600.0) < 1.0


def test_savitzky_golay_titresimi_azaltir():
    """Adım 13: hareketsiz başın titreşimi büyük sahte ivme üretmemeli."""
    rng = np.random.default_rng(0)
    xy = np.full((39, 2), 500.0) + rng.normal(0, 3, (39, 2))
    t = trajectory(xy)
    k = KinematicAnalyzer(30)
    new_max = k.compute_acceleration(t).max()
    # Eski yöntem: 3 karelik ortalama + iki kez fark
    sm = np.array([xy[max(0, i - 1):i + 2].mean(0) for i in range(len(xy))])
    v = np.linalg.norm(np.diff(sm, axis=0), axis=1) * 30
    old_max = np.abs(np.diff(v) * 30).max()
    assert new_max < old_max


def test_tepe_karesi_kare_numarasi_ve_pencere_icinde():
    """Adım 15: temas anı verilince ivme tepesi o anın ±0.5 sn penceresinde aranmalı."""
    xy = [(500, 300)] * 40
    for i in range(5, 8):        # klibin başında büyük bir hareket (pencere dışı)
        xy[i] = (500 + 80 * (i - 4), 300)
    for i in range(30, 33):      # temas anının yakınında küçük bir hareket
        xy[i] = (500 + 20 * (i - 29), 300)
    traj = [(x, y, f + 100) for f, (x, y) in enumerate(xy)]   # kareler 100'den başlıyor
    r = KinematicAnalyzer(30).classify_motion(traj, anchor_frame=131)
    assert r["anchor_frame"] == 131
    assert 131 - 15 <= r["peak_frame"] <= 131 + 15
    assert r["profile_start_frame"] == 100


def test_kisa_sure_ardisik_kareleri_sayar():
    """Adım 15 (madde 28): eşiği geçen kareler ardışık değilse süre en uzun kesintisiz parça olmalı."""
    import re
    from src.config import ACCELERATION_IMPACT_THRESHOLD

    k = KinematicAnalyzer(30)
    xy = [(500, 300)] * 60
    for i in (10, 25, 40):       # birbirinden uzak üç tekil sarsıntı
        xy[i] = (540, 300)
    t = trajectory(xy)
    r = k.classify_motion(t)
    duration = int(re.search(r"Süre: (\d+) frame", r["details"]).group(1))
    total_high = int((k.compute_acceleration(t) > ACCELERATION_IMPACT_THRESHOLD * 0.5).sum())
    assert duration < total_high     # eski yöntem tüm kareleri (ardışık olmasa da) toplardı
    assert duration <= 5
