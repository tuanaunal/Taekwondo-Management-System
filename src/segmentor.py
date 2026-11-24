import cv2
import numpy as np
import logging
from ultralytics import YOLO

YOLO_MODEL_PATH = 'yolov8n-pose.pt'

# Sporcu adayi olmak icin kutu alaninin, karedeki en buyuk kisinin kutu alanina orani.
# Kameradan uzaktaki seyirci/antrenor gibi kucuk gorunen kisileri eler.
MIN_ATHLETE_AREA_RATIO = 0.35

# Kimlik kopmasi: sporcu darbeyle dondugunde/dustugunde ByteTrack ona yeni bir takip numarasi
# verebilir ve o an gogus koruyucusu gorunmedigi icin renk oyu da olmaz. Bu durumda sporcu,
# son goruldugu konuma en yakin kisi ile surdurulur. Izin verilen en buyuk kayma: son kutunun
# kosegen uzunlugunun bu orani (bir karede bir insanin yarim boydan fazla yer degistirmesi beklenmez).
REID_MAX_SHIFT_RATIO = 0.5
# Iki takip numarasi en az bu kadar karede BIRLIKTE gorunduyse farkli kisilerdir.
# 1-2 karelik birliktelik, YOLO'nun ayni kisiyi ust uste iki kutuyla tespit etmesinden
# kaynaklanabilir (orn. RH_07'de dusen sporcu); bu yuzden tek kare yeterli sayilmaz.
MIN_COOCCURRENCE_FRAMES = 3

# Iskelet noktasi guven esigi: YOLO'nun bu noktadan emin olma derecesi (0-1).
# Esigin altindaki noktalar (orn. hareket bulanikligindaki ayak bilegi) "yok" sayilir.
# 0.5, Ultralytics'in noktalari cizerken kullandigi varsayilan esiktir.
KEYPOINT_CONF_THRESHOLD = 0.5

# Gogus koruyucusu renk tespiti (HSV, OpenCV olcegi: H 0-180, S/V 0-255)
# Karar renk tonu (H) ve doygunluga (S) dayanir; parlaklik (V) icin sadece cok dusuk bir
# alt sinir vardir. Karartma/aydinlatma V'yi degistirir ama H ve S'yi neredeyse degistirmez.
COLOR_MIN_SATURATION = 80     # Beyaz/gri dobok ve cilt gibi soluk pikselleri eler
COLOR_MIN_VALUE = 30          # Tamamen siyah (renk tonu anlamsiz) pikselleri eler
COLOR_MIN_FRACTION = 0.05     # Baskin renk, govde bolgesinin en az bu oraninda olmali
RED_HUE_RANGES = [(0, 10), (160, 180)]
BLUE_HUE_RANGE = (100, 140)

# YOLOv8-Pose (COCO) bas noktalari: 0 burun, 1 sol goz, 2 sag goz, 3 sol kulak, 4 sag kulak
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]

# Bas yaricapi tahmini icin vucut oranlari (ortalama yetiskin olculeri):
#   bas genisligi ~15 cm, iki kulak arasi ~bas genisligi, iki goz arasi ~6.3 cm,
#   govde boyu (omuz ortasi - kalca ortasi) ~50 cm
HEAD_RADIUS_PER_EAR_DISTANCE = 0.55     # yaricap ~ kulaklar arasi mesafenin yarisi (+ kask payi)
HEAD_RADIUS_PER_EYE_DISTANCE = 1.2      # 15 / 6.3 / 2
HEAD_RADIUS_PER_TORSO_LENGTH = 0.17     # (15 / 2 + kask payi) / 50

# Ayak ucu tahmini. Kafa vurusunda (dollyo chagi vb.) ayak bilekten ileri uzatilmis olur ve
# kafaya ayagin ustu / parmak tarafi deger; bu bolge ayak bileginden ~12-15 cm ileridedir.
# Ayak ucu, diz -> ayak bilegi dogrultusu uzatilarak tahmin edilir.
#   kaval kemigi (diz - ayak bilegi) ~42 cm  ->  15 / 42 ~ 0.35
FOOT_TIP_PER_SHANK_LENGTH = 0.35
# YOLOv8-Pose (COCO): 13 sol diz, 14 sag diz, 15 sol ayak bilegi, 16 sag ayak bilegi
LEGS = [(13, 15), (14, 16)]

logger = logging.getLogger(__name__)

class EquipmentSegmentor:
    """
    YOLOv8-Pose ile sporculari tespit edip kask (bas) ve ayak konumlarini uretir.
    ByteTrack ile her kisiye kalici takip numarasi verilir; kirmizi/mavi sporcu
    video geneli renk oylamasina gore secilir.
    """
    def __init__(self):
        logger.info(f"YOLOv8 Pose modeli yukleniyor: {YOLO_MODEL_PATH}")
        self.model = YOLO(YOLO_MODEL_PATH)
        # Kalici kimlik: ByteTrack her kisiye video boyunca ayni takip numarasini verir.
        # Her takip numarasi icin renk oylari ve gorunurluk istatistigi tutulur.
        self.color_votes = {}   # track_id -> {"red": int, "blue": int}
        self.track_stats = {}   # track_id -> {"frames": int, "prominence": float, "seen": set}
        self._frame_counter = 0  # 1. geciste islenen kare sayisi
        self.red_track = None   # Video geneline gore secilen kirmizi sporcunun takip numarasi
        self.blue_track = None  # Video geneline gore secilen mavi sporcunun takip numarasi
        self._last_box = {"red": None, "blue": None}  # Sporcunun son goruldugu kutu (2. gecis)

    # ────────────────────────────────────
    # RENK TESPITI (gogus koruyucusu / hogu)
    # ────────────────────────────────────
    @staticmethod
    def _torso_color(frame: np.ndarray, pts: np.ndarray) -> str:
        """Omuz-kalca arasindaki govde bolgesinin baskin rengini doner: red / blue / unknown."""
        h, w = frame.shape[:2]
        # YOLOv8 omuzlar: 5 (Sol), 6 (Sag) | Kalca: 11 (Sol), 12 (Sag)
        if not (pts[5][0] > 0 and pts[11][0] > 0):
            return "unknown"
        torso_x1 = int(min(pts[5][0], pts[6][0])) if pts[6][0] > 0 else int(pts[5][0] - 20)
        torso_x2 = int(max(pts[5][0], pts[6][0])) if pts[6][0] > 0 else int(pts[5][0] + 20)
        # Sag omuz/kalca yoksa (0, 0) sadece sol taraf kullanilir; aksi halde kutu karenin tepesine uzanirdi
        torso_y1 = int(min(pts[5][1], pts[6][1])) if pts[6][0] > 0 else int(pts[5][1])
        torso_y2 = int(max(pts[11][1], pts[12][1])) if pts[12][0] > 0 else int(pts[11][1])
        torso_x1, torso_y1 = max(0, torso_x1), max(0, torso_y1)
        torso_x2, torso_y2 = min(w, torso_x2), min(h, torso_y2)
        if torso_x1 >= torso_x2 or torso_y1 >= torso_y2:
            return "unknown"

        hsv = cv2.cvtColor(frame[torso_y1:torso_y2, torso_x1:torso_x2], cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        chromatic = (sat >= COLOR_MIN_SATURATION) & (val >= COLOR_MIN_VALUE)
        is_red = np.zeros_like(chromatic)
        for lo, hi in RED_HUE_RANGES:
            is_red |= (hue >= lo) & (hue <= hi)
        is_blue = (hue >= BLUE_HUE_RANGE[0]) & (hue <= BLUE_HUE_RANGE[1])
        red_pixels = int(np.count_nonzero(chromatic & is_red))
        blue_pixels = int(np.count_nonzero(chromatic & is_blue))

        min_pixels = max(20, COLOR_MIN_FRACTION * hue.size)
        if red_pixels > blue_pixels and red_pixels >= min_pixels:
            return "red"
        if blue_pixels > red_pixels and blue_pixels >= min_pixels:
            return "blue"
        return "unknown"

    # ────────────────────────────────────
    # 1. GECIS: KISILERI TESPIT ET VE ISTATISTIK TOPLA
    # ────────────────────────────────────
    def detect_persons(self, frame: np.ndarray) -> list:
        """
        Karedeki kisileri YOLO + ByteTrack ile tespit eder.
        Her kisi icin takip numarasi, iskelet noktalari, kutu alani ve govde rengi doner;
        ayrica video geneli renk oylarini ve gorunurluk istatistigini gunceller.
        """
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return []

        keypoints = results[0].keypoints.xy.cpu().numpy()          # [N, 17, 2]
        boxes = results[0].boxes.xyxy.cpu().numpy()                # [N, 4]
        if results[0].keypoints.conf is not None:
            confs = results[0].keypoints.conf.cpu().numpy()         # [N, 17]
        else:
            confs = np.ones(keypoints.shape[:2], dtype=np.float32)
        ids = results[0].boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(keypoints)

        persons = []
        for pts, conf, box, track_id in zip(keypoints, confs, boxes, track_ids):
            if len(pts) < 17:
                continue
            # Renk tespiti icin govde kutusu sadece guvenilir omuz/kalca noktalarindan kurulur;
            # guveni dusuk nokta (0, 0) yapilir ve _torso_color onu "yok" sayar.
            # Not: Bas ve ayak noktalari burada filtrelenmez. Dusuk guvenli bir noktayi atinca
            # yerine konacak bir sey olmadigi icin kafa/ayak o karede tamamen kayboluyordu
            # (20 videoda 13 -> 10). Bu noktalarin guveni, yerine diger bas noktalarinin ve
            # diz-ayak bilegi dogrultusunun kullanilabilecegi 8. ve 9. adimlarda kullanilacak.
            torso_pts = pts.copy()
            torso_pts[conf < KEYPOINT_CONF_THRESHOLD] = 0.0
            persons.append({
                "pts": pts,
                "conf": conf,
                "box": box,
                "track_id": track_id,
                "area": float((box[2] - box[0]) * (box[3] - box[1])),
                "color": self._torso_color(frame, torso_pts),
            })

        frame_no = self._frame_counter
        self._frame_counter += 1
        max_area = max((p["area"] for p in persons), default=0.0)
        for p in persons:
            if p["track_id"] is None:
                continue
            stats = self.track_stats.setdefault(
                p["track_id"], {"frames": 0, "prominence": 0.0, "seen": set()})
            stats["frames"] += 1
            stats["seen"].add(frame_no)  # Bu kisinin gorundugu kareler (kimlik kopmasi kontrolu icin)
            # Gorunurluk: kisinin bu karedeki boyutu, karedeki en buyuk kisiye gore (0-1)
            stats["prominence"] += p["area"] / max_area if max_area > 0 else 0.0
            if p["color"] != "unknown":
                votes = self.color_votes.setdefault(p["track_id"], {"red": 0, "blue": 0})
                votes[p["color"]] += 1
        return persons

    # ────────────────────────────────────
    # SPORCU SECIMI (video geneline gore, bir kez)
    # ────────────────────────────────────
    def _athlete_score(self, track_id, color: str) -> float:
        """
        Bir takip numarasinin verilen renkteki sporcu olma puani:
            gorunurluk x renk saflik orani
        Gorunurluk: video boyunca ne kadar buyuk ve ne kadar sik gorundugu.
        Saflik: (bu renk oyu - diger renk oyu) / gorundugu kare sayisi.
        Arka plandaki kucuk kisiler gorunurlukten, hakem (renksiz) saflik oranindan dusuk puan alir.
        """
        stats = self.track_stats.get(track_id)
        votes = self.color_votes.get(track_id)
        if not stats or not votes:
            return 0.0
        other = "blue" if color == "red" else "red"
        purity = max(0.0, (votes[color] - votes[other]) / stats["frames"])
        return stats["prominence"] * purity

    def select_athletes(self):
        """
        Video geneli istatistiklere gore kirmizi ve mavi sporcuyu secer.
        Eski yontemin aksine "en buyuk iki kutu" ve "soldaki kirmizidir" varsayimlari yoktur.
        """
        tracks = list(self.track_stats.keys())
        options = []
        for first in ("red", "blue"):
            second = "blue" if first == "red" else "red"
            a = max(tracks, key=lambda t: self._athlete_score(t, first), default=None)
            if a is not None and self._athlete_score(a, first) <= 0:
                a = None
            rest = [t for t in tracks if t != a]
            b = max(rest, key=lambda t: self._athlete_score(t, second), default=None)
            if b is not None and self._athlete_score(b, second) <= 0:
                b = None
            pair = {first: a, second: b}
            total = sum(self._athlete_score(pair[c], c) for c in pair if pair[c] is not None)
            options.append((total, pair))
        pair = max(options, key=lambda o: o[0])[1]
        self.red_track, self.blue_track = pair["red"], pair["blue"]
        logger.info(f"Secilen sporcular: kirmizi id={self.red_track}, mavi id={self.blue_track}")

    def _seen_with(self, track_a, track_b) -> bool:
        """Iki takip numarasi yeterince karede birlikte gorundu mu? (oyleyse farkli kisilerdir)"""
        a = self.track_stats.get(track_a, {}).get("seen", set())
        b = self.track_stats.get(track_b, {}).get("seen", set())
        return len(a & b) >= MIN_COOCCURRENCE_FRAMES

    def _pick_person(self, persons: list, color: str, exclude=None):
        """
        Secilen sporcuyu bu karede bulur.

        1. Sporcunun takip numarasi bu karede varsa o kisi secilir.
        2. Yoksa (kimlik koptuysa), sporcunun son goruldugu konuma en yakin kisi secilir.
           Darbe alip donen/dusen sporcu tam bu durumdadir: yeni takip numarasi alir ve o an
           gogus koruyucusu gorunmedigi icin renk oyu yoktur.
           Aday olamayanlar: rakip sporcunun parcasi (diger renkte oyu olan kisi) ve sporcuyla
           AYNI KAREDE gorunmus herkes (orn. arkadaki gorevli, hakem) - ayni anda ekranda
           olan biri o sporcu olamaz.
        3. Son konum bilinmiyorsa, video geneli puani bu renge uygun en buyuk kisi secilir
           (yine sporcuyla ayni karede gorunmus kisiler haric).
        Uygun kimse yoksa sporcu bu karede "yok" sayilir: yanlis kisiyi secmek (orn. hakem),
        sporcuyu bir kare kaybetmekten cok daha zararlidir.
        """
        target = self.red_track if color == "red" else self.blue_track
        other = "blue" if color == "red" else "red"
        chosen = None
        if target is not None:
            chosen = next((p for p in persons if p["track_id"] == target), None)

        if chosen is None and self._last_box[color] is not None:
            lx1, ly1, lx2, ly2 = self._last_box[color]
            last_center = np.array([(lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0])
            max_shift = REID_MAX_SHIFT_RATIO * float(np.hypot(lx2 - lx1, ly2 - ly1))
            nearby = []
            for p in persons:
                if p is exclude or p["track_id"] in (self.red_track, self.blue_track):
                    continue
                if self._athlete_score(p["track_id"], other) > 0:
                    continue
                if self._seen_with(p["track_id"], target):
                    continue  # Sporcuyla ayni anda ekranda gorunmus: baska bir kisi
                x1, y1, x2, y2 = p["box"]
                shift = float(np.linalg.norm(np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]) - last_center))
                if shift <= max_shift:
                    nearby.append((shift, p))
            if nearby:
                chosen = min(nearby, key=lambda t: t[0])[1]

        if chosen is None:
            chosen = self._pick_by_color(persons, color, exclude, target)

        if chosen is not None:
            self._last_box[color] = chosen["box"]
        return chosen

    def _pick_by_color(self, persons: list, color: str, exclude=None, target=None):
        """Yedek secim: video geneli puani bu renge uygun en buyuk kisi."""
        max_area = max((p["area"] for p in persons), default=0.0)
        candidates = [
            p for p in persons
            if p is not exclude
            and p["track_id"] not in (self.red_track, self.blue_track)
            and not (target is not None and self._seen_with(p["track_id"], target))
            and p["area"] >= MIN_ATHLETE_AREA_RATIO * max_area
            and (self._athlete_score(p["track_id"], color) > 0
                 or (p["track_id"] is None and p["color"] == color))
        ]
        return max(candidates, key=lambda p: p["area"], default=None)

    # ────────────────────────────────────
    # BAS (KASK) BOLGESI
    # ────────────────────────────────────
    @staticmethod
    def _head_center(pts: np.ndarray, conf: np.ndarray):
        """
        Bas merkezini burun, goz ve kulak noktalarindan hesaplar.

        - Guvenilir (esigin ustundeki) bas noktalarinin guvene gore agirlikli ortalamasi alinir.
          Sporcu basini cevirdiginde burun kaybolsa bile kulaklardan bas konumu bulunur.
        - Hicbir bas noktasi guvenilir degilse, bilgi kaybolmasin diye gorunen en yuksek
          guvenli bas noktasi kullanilir (eskiden kullanilan tek burun noktasina esdeger).
        """
        visible = [i for i in HEAD_KEYPOINTS if pts[i][0] > 0 and pts[i][1] > 0]
        if not visible:
            return None
        reliable = [i for i in visible if conf[i] >= KEYPOINT_CONF_THRESHOLD]
        if reliable:
            weights = np.array([conf[i] for i in reliable], dtype=float)
            center = np.average(np.array([pts[i] for i in reliable], dtype=float), axis=0, weights=weights)
        else:
            center = pts[max(visible, key=lambda i: conf[i])]
        return (int(center[0]), int(center[1]))

    @staticmethod
    def _head_radius(pts: np.ndarray, conf: np.ndarray):
        """
        Bas yaricapini (piksel) vucut oranlarindan tahmin eder.

        Uc tahmin yapilir: iki kulak arasi, iki goz arasi ve govde boyu (omuz ortasi - kalca
        ortasi). Sporcu yana dondugunde kulaklar/gozler ust uste biner ve bu mesafeler
        kisalir; yana donmek mesafeleri sadece kisaltabilir, uzatamaz. Bu yuzden mevcut
        tahminlerin EN BUYUGU alinir. Hicbiri hesaplanamazsa None doner.
        """
        def ok(i):
            return conf[i] >= KEYPOINT_CONF_THRESHOLD and pts[i][0] > 0 and pts[i][1] > 0

        def dist(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        estimates = []
        if ok(3) and ok(4):
            estimates.append(dist(pts[3], pts[4]) * HEAD_RADIUS_PER_EAR_DISTANCE)
        if ok(1) and ok(2):
            estimates.append(dist(pts[1], pts[2]) * HEAD_RADIUS_PER_EYE_DISTANCE)
        if ok(5) and ok(6) and ok(11) and ok(12):
            shoulder_mid = (pts[5] + pts[6]) / 2.0
            hip_mid = (pts[11] + pts[12]) / 2.0
            estimates.append(dist(shoulder_mid, hip_mid) * HEAD_RADIUS_PER_TORSO_LENGTH)
        return max(estimates) if estimates else None

    # ────────────────────────────────────
    # AYAK UCU
    # ────────────────────────────────────
    @staticmethod
    def _foot_tips(pts: np.ndarray, conf: np.ndarray) -> list:
        """
        Her bacak icin ayagin kafaya degen kismini (ayak ustu / parmak tarafi) tahmin eder.

        - Diz ve ayak bilegi gorunuyorsa: ayak ucu = ayak bilegi + 0.35 x (ayak bilegi - diz)
        - Sadece ayak bilegi gorunuyorsa: ayak bilegi kullanilir (eski davranis, bilgi kaybolmaz)
        Guveni dusuk diz, dogrultuyu bozmamak icin kullanilmaz; guveni dusuk ayak bilegi ise
        yerine konacak bir sey olmadigi icin yine kullanilir.
        """
        def visible(i):
            return pts[i][0] > 0 and pts[i][1] > 0

        tips = []
        for knee, ankle in LEGS:
            if not visible(ankle):
                continue
            a = np.asarray(pts[ankle], dtype=float)
            if visible(knee) and conf[knee] >= KEYPOINT_CONF_THRESHOLD:
                k = np.asarray(pts[knee], dtype=float)
                tip = a + FOOT_TIP_PER_SHANK_LENGTH * (a - k)
            else:
                tip = a
            tips.append((int(tip[0]), int(tip[1])))
        return tips

    # ────────────────────────────────────
    # 2. GECIS: SECILEN SPORCULARDAN KASK / AYAK KONUMLARINI URET
    # ────────────────────────────────────
    def build_features(self, persons: list, h: int, w: int) -> dict:
        """Secilen sporcularin kask ve ayak konumlarindan temas analizi icin ozellikleri uretir."""
        red_person = self._pick_person(persons, "red")
        blue_person = self._pick_person(persons, "blue", exclude=red_person)

        def get_all_feet(pts):
            ankle_left, ankle_right = pts[15], pts[16]
            ankles = [p for p in [ankle_left, ankle_right] if p[0] > 0 and p[1] > 0]
            return [(int(p[0]), int(p[1])) for p in ankles]

        def smooth_pt(new_pt, attr_name, max_jump=150, max_hold=10):
            hist_pt = getattr(self, f"{attr_name}_hist", None)
            lost_frames = getattr(self, f"{attr_name}_lost", 0)

            if new_pt is not None and hist_pt is not None:
                dist = np.hypot(new_pt[0] - hist_pt[0], new_pt[1] - hist_pt[1])
                if dist > max_jump:
                    new_pt = None # Ani atlama reddedildi

            if new_pt is not None:
                setattr(self, f"{attr_name}_hist", new_pt)
                setattr(self, f"{attr_name}_lost", 0)
                return new_pt
            else:
                if hist_pt is not None and lost_frames < max_hold:
                    setattr(self, f"{attr_name}_lost", lost_frames + 1)
                    return hist_pt
                return None

        def nose_point(pts):
            if pts[0][0] > 0 and pts[0][1] > 0:
                return (int(pts[0][0]), int(pts[0][1]))
            return None

        # Kask noktasi (hareket analizi icin) simdilik BURUN olarak kalir.
        # Olcum: gercek darbede burun, bas merkezinden belirgin sekilde fazla hareket ediyor
        # (Real Hit savrulma ortancasi: burun 140 px, bas merkezi 104 px; Ghost Hit'te ikisi ~100 px).
        # Sebep, darbenin basi dondurmesi: burun basin onunde oldugu icin donuste yay cizer.
        # Bas merkezine gecmek bu bilgiyi siliyordu (20 videoda 12 -> 10). Bas donusu 14. adimda
        # ayri bir ozellik olarak eklenecek; bas merkezi ve yaricap 12. adimda temas icin kullanilacak.
        raw_red_helm = None
        raw_red_feet = []
        raw_blue_helm = None
        raw_blue_feet = []
        red_head_center = red_head_radius = None
        blue_head_center = blue_head_radius = None
        red_foot_tips = []
        blue_foot_tips = []

        if red_person:
            pts, conf = red_person["pts"], red_person["conf"]
            raw_red_helm = nose_point(pts)
            red_head_center = self._head_center(pts, conf)
            red_head_radius = self._head_radius(pts, conf)
            raw_red_feet = get_all_feet(pts)
            red_foot_tips = self._foot_tips(pts, conf)

        if blue_person:
            pts, conf = blue_person["pts"], blue_person["conf"]
            raw_blue_helm = nose_point(pts)
            blue_head_center = self._head_center(pts, conf)
            blue_head_radius = self._head_radius(pts, conf)
            raw_blue_feet = get_all_feet(pts)
            blue_foot_tips = self._foot_tips(pts, conf)

        # Sadece Kasklar icin ziplama filtresi kullan (Kafalar 200 piksellik isinlanma yapamaz)
        # Ayaklar ise saniyede cok hizli hareket ettigi (ucan tekme) icin filtrelemeden direkt gonderiyoruz!
        red_helmet_pt = smooth_pt(raw_red_helm, "red_helm", max_jump=200, max_hold=10)
        blue_helmet_pt = smooth_pt(raw_blue_helm, "blue_helm", max_jump=200, max_hold=10)

        output = {}
        # Masks: 80px radius (160px total reach to catch overlapping hits)
        output["red_helmet"] = self._create_fake_features([red_helmet_pt] if red_helmet_pt else [], h, w, radius=80)
        output["red_foot"] = self._create_fake_features(raw_red_feet, h, w, radius=80)
        output["blue_helmet"] = self._create_fake_features([blue_helmet_pt] if blue_helmet_pt else [], h, w, radius=80)
        output["blue_foot"] = self._create_fake_features(raw_blue_feet, h, w, radius=80)
        # Bas merkezi (5 bas noktasindan) ve tahmini bas yaricapi (piksel).
        # Temas olcumu bunlari 12. adimda kullanacak; o zamana kadar maskeler eski
        # yontemle (burun + 80 px yaricap) uretilmeye devam ediyor.
        output["red_helmet"]["head_center"] = red_head_center
        output["red_helmet"]["head_radius"] = red_head_radius
        output["blue_helmet"]["head_center"] = blue_head_center
        output["blue_helmet"]["head_radius"] = blue_head_radius
        # Tahmini ayak uclari (ayak ustu / parmak tarafi). Temas olcumu 12. adimda bunlari
        # kullanacak; o zamana kadar ayak maskeleri ayak bileginden uretilmeye devam ediyor.
        output["red_foot"]["foot_tips"] = red_foot_tips
        output["blue_foot"]["foot_tips"] = blue_foot_tips

        return output

    def segment_video(self, frames):
        """
        Iki gecisli analiz: once tum karelerde kisiler tespit edilir, sonra sporcular video
        geneline gore bir kez secilir ve her kare icin ozellikler uretilir.

        Parameters
        ----------
        frames : iterable of np.ndarray
            On-islemden gecmis kareler.

        Returns
        -------
        iterator of dict
            Her kare icin build_features() ciktisi. Maskeler buyuk oldugu icin
            hepsi ayni anda bellekte tutulmaz, sirayla uretilir.
        """
        detections = []
        h = w = 0
        for frame in frames:
            h, w = frame.shape[:2]
            detections.append(self.detect_persons(frame))
        self.select_athletes()
        return (self.build_features(persons, h, w) for persons in detections)

    def segment_frame(self, frame: np.ndarray) -> dict:
        """
        Tek kare (cevrimici) kullanim icin: sporcu secimi o ana kadar gorulen karelere gore yapilir.
        Video analizi icin segment_video() tercih edilmelidir.
        """
        h, w = frame.shape[:2]
        persons = self.detect_persons(frame)
        self.select_athletes()
        return self.build_features(persons, h, w)

    def _create_fake_features(self, pts_list, h, w, radius):
        mask = np.zeros((h, w), dtype=np.uint8)
        contours = []
        centroids = []
        
        if not isinstance(pts_list, list):
            pts_list = [pts_list]
            
        for pt in pts_list:
            if pt is not None and (pt[0] > 0 or pt[1] > 0):
                cv2.circle(mask, pt, radius, 255, -1)
                # Dairesel bir contour oluştur
                pts = []
                for angle in range(0, 360, 10):
                    x = int(pt[0] + radius * np.cos(np.radians(angle)))
                    y = int(pt[1] + radius * np.sin(np.radians(angle)))
                    if 0 <= x < w and 0 <= y < h:
                        pts.append([[x, y]])
                if pts:
                    contours.append(np.array(pts, dtype=np.int32))
                centroids.append(pt)
                
        return {"mask": mask, "contours": contours, "centroids": centroids}

    def draw_segmentation(self, frame: np.ndarray, seg_results: dict, alpha: float = 0.5) -> np.ndarray:
        """GUI icin segmentasyon maskelerini ve centroidleri cizer."""
        overlay = frame.copy()
        fh, fw = frame.shape[:2]

        colors = {
            "red_helmet": (0, 0, 255),
            "red_foot": (0, 0, 255),
            "blue_helmet": (255, 0, 0),
            "blue_foot": (255, 0, 0)
        }
        
        for key, color in colors.items():
            if key in seg_results:
                mask = seg_results[key].get("mask")
                # Maske, on-islemde buyutulmus kareye gore uretilir (orn. 1920x1080).
                # Cizilen kare farkli boyuttaysa koordinatlari o boyuta olcekle.
                sx, sy = 1.0, 1.0
                if mask is not None:
                    mh, mw = mask.shape[:2]
                    sx, sy = fw / mw, fh / mh
                    if (mh, mw) != (fh, fw):
                        mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask > 0] = color
                    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

                centroids = seg_results[key].get("centroids", [])
                for cx, cy in centroids:
                    cv2.circle(overlay, (int(cx * sx), int(cy * sy)), 5, (0, 255, 0), -1)
                    
        return overlay
