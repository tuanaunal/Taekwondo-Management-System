"""
config.py — Taekwondo Ghost Hit Detection: Merkezi Konfigürasyon Modülü
=======================================================================
Projenin tüm ayarlarını (HSV renk aralıkları, kinematik eşikler,
dosya yolları vb.) tek bir noktadan yönetir.
"""

import os
import numpy as np

# ──────────────────────────────────────────────
# 1. DOSYA YOLLARI
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
GHOST_HIT_DIR = os.path.join(DATASET_DIR, "Ghost_Hit")
REAL_HIT_DIR = os.path.join(DATASET_DIR, "Real_Hit")
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "Raw_Videos")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Çıktı dizinlerini oluştur
for d in [LOGS_DIR, GRAPHS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# 2. VIDEO STANDARDIZASYON
# ──────────────────────────────────────────────
TARGET_FPS = 30
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# ──────────────────────────────────────────────
# 3. CLAHE ve ÖN-İŞLEME PARAMETRELERİ
# ──────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

GAUSSIAN_KERNEL_SIZE = (5, 5)
MEDIAN_KERNEL_SIZE = 5

# ──────────────────────────────────────────────
# 4. HSV RENK ARALLIKLARI (Taekwondo Ekipmanları)
# ──────────────────────────────────────────────
# Kırmızı renk HSV uzayında 0° ve 180° civarında iki parçaya ayrılır.
# Her aralık (H_min, S_min, V_min) ve (H_max, S_max, V_max) olarak tanımlanır.

HSV_RANGES = {
    # ── KIRMIZI KASK (Hogu) ──
    "red_helmet_lower1": np.array([0, 30, 20]),
    "red_helmet_upper1": np.array([12, 255, 255]),
    "red_helmet_lower2": np.array([165, 30, 20]),
    "red_helmet_upper2": np.array([180, 255, 255]),

    # ── MAVİ KASK (Hogu) ──
    "blue_helmet_lower": np.array([90, 30, 20]),
    "blue_helmet_upper": np.array([130, 255, 255]),

    # ── KIRMIZI AYAK KORUYUCU ──
    "red_foot_lower1": np.array([0, 30, 20]),
    "red_foot_upper1": np.array([12, 255, 255]),
    "red_foot_lower2": np.array([165, 30, 20]),
    "red_foot_upper2": np.array([180, 255, 255]),

    # ── MAVİ AYAK KORUYUCU ──
    "blue_foot_lower": np.array([90, 30, 20]),
    "blue_foot_upper": np.array([130, 255, 255]),
}

# ──────────────────────────────────────────────
# 5. MORFOLOJİK OPERASYON PARAMETRELERİ
# ──────────────────────────────────────────────
MORPH_KERNEL_SIZE = (7, 7)
MORPH_ERODE_ITERATIONS = 1
MORPH_DILATE_ITERATIONS = 2
MORPH_CLOSE_ITERATIONS = 2

# Minimum kontur alanı (piksel²) — küçük gürültüleri filtrele
MIN_CONTOUR_AREA = 500

# Maksimum kontur alanı (piksel²) — devasa sahte tespitleri filtrele
# Gerçek kask: ~15,000-30,000 px², gerçek ayak: ~3,000-20,000 px²
# Bu değerin üzerindeki konturlar zemin/arka plan olarak kabul edilir
MAX_CONTOUR_AREA = 350000

# ──────────────────────────────────────────────
# 6. KİNEMATİK ANALİZ EŞİKLERİ
# ──────────────────────────────────────────────
# İvme eşik değeri (piksel/frame²)
# Bu değerin üzerindeki ani ivme artışları "darbe" olarak sınıflandırılır
# NOT: kinematics.py ivmeyi piksel/saniye² cinsinden hesapladığından, 15.0 px/frame² eşiği
# 15 * 30^2 = 13500 px/s² civarına denk gelir. Gürültü ve gerçek tekmeleri dengelemek için 8000.0 px/s² seçilmiştir.
ACCELERATION_IMPACT_THRESHOLD = 8000.0

# İvme profili sınıflandırma parametreleri
# Darbe ivmesi: kısa sürede yüksek tepe noktası
# Aktif kaçış: kademeli, düşük ivme
IMPACT_DURATION_MAX_FRAMES = 5       # Darbe en fazla bu kadar frame sürer
EVASION_SMOOTHNESS_THRESHOLD = 0.6   # İvme varyasyon katsayısı eşiği

# ──────────────────────────────────────────────
# 7. TEMAS ANALİZİ EŞİKLERİ
# ──────────────────────────────────────────────
# Piksel bazlı çakışma eşiği — bu kadar piksel çakışırsa temas VAR
# Not: Segmentasyon düzeltildikten sonra 20px yeterli hassasiyet sağlıyor
CONTACT_OVERLAP_THRESHOLD = 20

# Öklid mesafesi eşiği (piksel) — merkezler arası
# Bu mesafeden yakınsa "yakın geçiş" (near miss), değilse "temas yok"
PROXIMITY_THRESHOLD = 80

# Kontur sınır mesafesi eşiği (piksel)
# 20px içindeyse "yakın geçiş" (near miss)
CONTOUR_DISTANCE_THRESHOLD = 20

# ──────────────────────────────────────────────
# 8. ROI (Region of Interest) AYARLARI
# ──────────────────────────────────────────────
# Arka plan çıkarma (MOG2) parametreleri
MOG2_HISTORY = 50
MOG2_VAR_THRESHOLD = 40
MOG2_DETECT_SHADOWS = False

# ROI genişletme marjini (piksel)
ROI_MARGIN = 50

# ──────────────────────────────────────────────
# 9. GÖRSELLEŞTİRME AYARLARI
# ──────────────────────────────────────────────
# Renk kodları (BGR formatı, OpenCV için)
VIS_COLOR_RED = (0, 0, 255)
VIS_COLOR_BLUE = (255, 120, 0)
VIS_COLOR_GREEN = (0, 255, 0)
VIS_COLOR_YELLOW = (0, 255, 255)
VIS_COLOR_WHITE = (255, 255, 255)

# Çizgi kalınlıkları
VIS_LINE_THICKNESS = 2
VIS_FONT_SCALE = 0.7

# ──────────────────────────────────────────────
# 10. KARAR MOTORU AĞIRLIKLARI
# ──────────────────────────────────────────────
# Nihai karar için bileşen ağırlıkları
WEIGHT_CONTACT = 0.5        # Temas analizi ağırlığı
WEIGHT_KINEMATICS = 0.5     # Kinematik analiz ağırlığı

# Güven skoru eşiği
CONFIDENCE_THRESHOLD = 0.6
