"""Sporcu seçimi ve iskelet yardımcıları (YOLO modeli yüklenmeden)."""
import numpy as np

from src.segmentor import EquipmentSegmentor, MIN_COOCCURRENCE_FRAMES


def segmentor_without_model():
    """YOLO'yu yüklemeden yardımcı fonksiyonları test etmek için boş bir segmentor."""
    s = object.__new__(EquipmentSegmentor)
    s.color_votes, s.track_stats = {}, {}
    s.red_track = s.blue_track = None
    s._last_box = {"red": None, "blue": None}
    return s


def pose(points: dict, conf=0.9):
    pts = np.zeros((17, 2), dtype=float)
    c = np.zeros(17, dtype=float)
    for i, p in points.items():
        pts[i] = p
        c[i] = conf
    return pts, c


def test_bas_yaricapi_yan_donuste_kuculmez():
    """Adım 8: kulaklar üst üste binse de gövde boyundan makul bir yarıçap bulunmalı."""
    pts, conf = pose({3: (500, 100), 4: (501, 100),                    # yan dönüş: kulaklar üst üste
                      5: (470, 150), 6: (530, 150), 11: (475, 300), 12: (525, 300)})
    r = EquipmentSegmentor._head_radius(pts, conf)
    assert r is not None and r > 20      # gövde boyu 150 px * 0.17 = ~25 px


def test_ayak_ucu_diz_bilek_dogrultusunda():
    """Adım 9: ayak ucu = bilek + 0.35 x (bilek - diz)."""
    pts, conf = pose({13: (500, 300), 15: (500, 400)})
    tips = EquipmentSegmentor._foot_tips(pts, conf)
    assert tips == [(500, 435)]


def test_renk_karanlikta_da_okunur():
    """Adım 7: kırmızı göğüs koruyucusu karartılmış görüntüde de kırmızı okunmalı."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[150:300, 160:240] = (30, 30, 200)                              # BGR kırmızı
    pts, _ = pose({5: (160, 150), 6: (240, 150), 11: (165, 300), 12: (235, 300)})
    assert EquipmentSegmentor._torso_color(img, pts) == "red"
    dark = (img * 0.35).astype(np.uint8)
    assert EquipmentSegmentor._torso_color(dark, pts) == "red"


def test_birlikte_gorunen_kisi_ayni_sporcu_olamaz():
    """Adım 9: en az MIN_COOCCURRENCE_FRAMES karede birlikte görünen iki kimlik farklı kişidir."""
    s = segmentor_without_model()
    s.track_stats = {1: {"seen": {0, 1, 2, 3}}, 2: {"seen": {3}}, 3: {"seen": {0, 1, 2}}}
    assert not s._seen_with(1, 2)        # tek karelik çift tespit tolere edilir
    assert s._seen_with(1, 3) == (3 >= MIN_COOCCURRENCE_FRAMES)


def test_sporcu_secimi_gorunurluk_ve_renk_tutarliligi():
    """Adım 6: renksiz hakem ve arkadaki küçük kişi değil, büyük ve renk tutarlı kişiler seçilir."""
    s = segmentor_without_model()
    s.track_stats = {
        1: {"frames": 30, "prominence": 28.0, "seen": set()},   # kırmızı sporcu
        2: {"frames": 30, "prominence": 25.0, "seen": set()},   # mavi sporcu
        3: {"frames": 30, "prominence": 27.0, "seen": set()},   # hakem (renksiz)
        4: {"frames": 30, "prominence": 3.0, "seen": set()},    # arkada küçük, mavi giyimli seyirci
    }
    s.color_votes = {1: {"red": 28, "blue": 1}, 2: {"red": 0, "blue": 27},
                     3: {"red": 1, "blue": 1}, 4: {"red": 0, "blue": 30}}
    s.select_athletes()
    assert (s.red_track, s.blue_track) == (1, 2)
