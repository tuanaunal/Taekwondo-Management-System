import cv2
import numpy as np
import logging
from ultralytics import YOLO

YOLO_MODEL_PATH = 'yolov8n-pose.pt'

logger = logging.getLogger(__name__)

class EquipmentSegmentor:
    """
    YOLOv8-Pose kullanarak Kask ve Ayak konumlarını tespit eder.
    Hatalı (0,0) konumlarını önlemek için akıllı fallback (Bilek -> Diz -> Kalça) mekanizması kullanır.
    """
    def __init__(self):
        logger.info(f"YOLOv8 Pose modeli yukleniyor: {YOLO_MODEL_PATH}")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.last_red_x = None

    def segment_frame(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        
        # 1. YOLO ile tahmin
        results = self.model(frame, verbose=False)
        
        red_helmet_pt = None
        red_foot_pt = None
        blue_helmet_pt = None
        blue_foot_pt = None
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy() # [N, 17, 2]
            boxes = results[0].boxes.xyxy.cpu().numpy() # [N, 4]
            
            persons = []
            for i, pts in enumerate(keypoints):
                if len(pts) < 17: continue
                box = boxes[i]
                center_x = (box[0] + box[2]) / 2
                
                # Sirt/Gövde bölgesindeki renk ağırlığını kontrol edelim
                # YOLOv8 omuzlar: 5 (Sol), 6 (Sağ) | Kalça: 11 (Sol), 12 (Sağ)
                color_label = "unknown"
                if pts[5][0] > 0 and pts[11][0] > 0:
                    torso_x1 = int(min(pts[5][0], pts[6][0])) if pts[6][0] > 0 else int(pts[5][0] - 20)
                    torso_x2 = int(max(pts[5][0], pts[6][0])) if pts[6][0] > 0 else int(pts[5][0] + 20)
                    torso_y1 = int(min(pts[5][1], pts[6][1]))
                    torso_y2 = int(max(pts[11][1], pts[12][1]))
                    
                    if torso_x1 < torso_x2 and torso_y1 < torso_y2:
                        torso_x1, torso_y1 = max(0, torso_x1), max(0, torso_y1)
                        torso_x2, torso_y2 = min(w, torso_x2), min(h, torso_y2)
                        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
                        if torso_crop.size > 0:
                            hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                            # Kırmızı
                            mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                            mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                            red_pixels = cv2.countNonZero(cv2.bitwise_or(mask_red1, mask_red2))
                            # Mavi
                            mask_blue = cv2.inRange(hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))
                            blue_pixels = cv2.countNonZero(mask_blue)
                            
                            if red_pixels > blue_pixels and red_pixels > 20:
                                color_label = "red"
                            elif blue_pixels > red_pixels and blue_pixels > 20:
                                color_label = "blue"
                
                persons.append({"pts": pts, "color": color_label, "center_x": center_x})
                
            # Eğer renk bulamadıysa, son kırmızı pozisyonuna göre ayır
            if len(persons) == 2:
                colors = [p["color"] for p in persons]
                if colors.count("unknown") > 0:
                    persons.sort(key=lambda p: p["center_x"])
                    if self.last_red_x is not None:
                        dist0 = abs(persons[0]["center_x"] - self.last_red_x)
                        dist1 = abs(persons[1]["center_x"] - self.last_red_x)
                        if dist0 < dist1:
                            persons[0]["color"] = "red"
                            persons[1]["color"] = "blue"
                        else:
                            persons[0]["color"] = "blue"
                            persons[1]["color"] = "red"
                    else:
                        persons[0]["color"] = "red"
                        persons[1]["color"] = "blue"
                            
            red_person = next((p for p in persons if p["color"] == "red"), None)
            blue_person = next((p for p in persons if p["color"] == "blue"), None)

            def get_best_foot(pts):
                ankle_left, ankle_right = pts[15], pts[16]
                knee_left, knee_right = pts[13], pts[14]
                
                # 1. Try Ankles (highest one)
                ankles = [p for p in [ankle_left, ankle_right] if p[0] > 0 and p[1] > 0]
                if ankles:
                    ankles.sort(key=lambda p: p[1])
                    return (int(ankles[0][0]), int(ankles[0][1])), "ankle"
                    
                # We NO LONGER fallback to Knees or Hips!
                # Falling back to Knees/Hips creates fake "distances" that ruin proximity rules.
                # Instead, we will return None and let the history-holder keep the last position!
                return None, "none"

            if red_person:
                self.last_red_x = red_person["center_x"]
                pts = red_person["pts"]
                if pts[0][0] > 0 and pts[0][1] > 0:
                    red_helmet_pt = (int(pts[0][0]), int(pts[0][1]))
                
                foot_pt, label = get_best_foot(pts)
                if foot_pt:
                    red_foot_pt = foot_pt
                    self.red_foot_history = foot_pt
                    self.red_foot_lost_frames = 0
                else:
                    # Hold last known position for up to 5 frames (occlusion fix)
                    if hasattr(self, 'red_foot_history') and self.red_foot_lost_frames < 5:
                        red_foot_pt = self.red_foot_history
                        self.red_foot_lost_frames += 1
                    
            if blue_person:
                pts = blue_person["pts"]
                if pts[0][0] > 0 and pts[0][1] > 0:
                    blue_helmet_pt = (int(pts[0][0]), int(pts[0][1]))
                
                foot_pt, label = get_best_foot(pts)
                if foot_pt:
                    blue_foot_pt = foot_pt
                    self.blue_foot_history = foot_pt
                    self.blue_foot_lost_frames = 0
                else:
                    if hasattr(self, 'blue_foot_history') and self.blue_foot_lost_frames < 5:
                        blue_foot_pt = self.blue_foot_history
                        self.blue_foot_lost_frames += 1

        output = {}
        # Masks: 80px radius (160px total reach to catch overlapping hits)
        output["red_helmet"] = self._create_fake_features(red_helmet_pt, h, w, radius=80)
        output["red_foot"] = self._create_fake_features(red_foot_pt, h, w, radius=80)
        output["blue_helmet"] = self._create_fake_features(blue_helmet_pt, h, w, radius=80)
        output["blue_foot"] = self._create_fake_features(blue_foot_pt, h, w, radius=80)


        
        return output

    def _create_fake_features(self, pt, h, w, radius):
        mask = np.zeros((h, w), dtype=np.uint8)
        contours = []
        centroids = []
        
        # Eğer nokta None değilse ve (0,0) değilse çiz
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
