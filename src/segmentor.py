import cv2
import numpy as np
import logging
from ultralytics import YOLO

YOLO_MODEL_PATH = 'yolov8n-pose.pt'

# Sporcu adayi olmak icin kutu alaninin, karedeki en buyuk kisinin kutu alanina orani.
# Kameradan uzaktaki seyirci/antrenor gibi kucuk gorunen kisileri eler.
MIN_ATHLETE_AREA_RATIO = 0.35

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
        self.track_stats = {}   # track_id -> {"frames": int, "prominence": float}
        self.red_track = None   # Video geneline gore secilen kirmizi sporcunun takip numarasi
        self.blue_track = None  # Video geneline gore secilen mavi sporcunun takip numarasi

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
                "track_id": track_id,
                "area": float((box[2] - box[0]) * (box[3] - box[1])),
                "color": self._torso_color(frame, torso_pts),
            })

        max_area = max((p["area"] for p in persons), default=0.0)
        for p in persons:
            if p["track_id"] is None:
                continue
            stats = self.track_stats.setdefault(p["track_id"], {"frames": 0, "prominence": 0.0})
            stats["frames"] += 1
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

    def _pick_person(self, persons: list, color: str, exclude=None):
        """
        Secilen sporcuyu bu karede bulur. Sporcunun takip numarasi bu karede yoksa
        (ByteTrack kimligi kisa sureligine kopardiginda), video geneli puani bu renge
        uygun olan baska bir buyuk kisi yedek olarak kullanilir.
        """
        target = self.red_track if color == "red" else self.blue_track
        if target is not None:
            for p in persons:
                if p["track_id"] == target:
                    return p
        max_area = max((p["area"] for p in persons), default=0.0)
        candidates = [
            p for p in persons
            if p is not exclude
            and p["track_id"] not in (self.red_track, self.blue_track)
            and p["area"] >= MIN_ATHLETE_AREA_RATIO * max_area
            and (self._athlete_score(p["track_id"], color) > 0
                 or (p["track_id"] is None and p["color"] == color))
        ]
        return max(candidates, key=lambda p: p["area"], default=None)

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

        raw_red_helm = None
        raw_red_feet = []
        raw_blue_helm = None
        raw_blue_feet = []

        if red_person:
            pts = red_person["pts"]
            if pts[0][0] > 0 and pts[0][1] > 0:
                raw_red_helm = (int(pts[0][0]), int(pts[0][1]))
            raw_red_feet = get_all_feet(pts)

        if blue_person:
            pts = blue_person["pts"]
            if pts[0][0] > 0 and pts[0][1] > 0:
                raw_blue_helm = (int(pts[0][0]), int(pts[0][1]))
            raw_blue_feet = get_all_feet(pts)

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
