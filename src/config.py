"""
config.py — Taekwondo Hakem Karar Destek Sistemi: Merkezi Ayarlar
=================================================================
Projenin ortak ayarlarını (dosya yolları, ön işleme, kinematik ve temas
eşikleri) tek bir noktadan yönetir. Modüle özel ayarlar kendi dosyalarında durur
(ör. renk tespiti: segmentor.py, karar modeli: model/karar_modeli.json).
"""

import os

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
# Videonun FPS bilgisi okunamazsa kullanılan varsayılan değer.
# Kinematik hesaplar her zaman videonun GERÇEK FPS değeriyle yapılır.
TARGET_FPS = 30
# Not: Çözünürlük standardizasyonu (eski TARGET_WIDTH/HEIGHT) kaldırıldı;
# mesafeler sporcunun baş yarıçapına göre ölçekleniyor (bkz. segmentor.py).

# ──────────────────────────────────────────────
# 3. CLAHE ve ÖN-İŞLEME PARAMETRELERİ
# ──────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ──────────────────────────────────────────────
# 4. KİNEMATİK ANALİZ EŞİKLERİ
# ──────────────────────────────────────────────
# İvme eşik değeri (piksel/frame²)
# Bu değerin üzerindeki ani ivme artışları "darbe" olarak sınıflandırılır
# NOT: kinematics.py ivmeyi piksel/saniye² cinsinden hesapladığından, 15.0 px/frame² eşiği
# 15 * 30^2 = 13500 px/s² civarına denk gelir. Gürültü ve gerçek tekmeleri dengelemek için YOLO ile 1500.0 px/s² seçilmiştir.
ACCELERATION_IMPACT_THRESHOLD = 1500.0

# İvme profili sınıflandırma parametreleri
# Darbe ivmesi: kısa sürede yüksek tepe noktası
# Aktif kaçış: kademeli, düşük ivme
IMPACT_DURATION_MAX_FRAMES = 5       # Darbe en fazla bu kadar frame sürer (30 FPS referansıyla;
                                     # farklı FPS'te aynı süreye denk gelecek şekilde ölçeklenir)
EVASION_SMOOTHNESS_THRESHOLD = 0.6   # İvme varyasyon katsayısı eşiği

# ──────────────────────────────────────────────
# 5. TEMAS ANALİZİ EŞİKLERİ
# ──────────────────────────────────────────────
# Temas, ayak noktalarının (ayak ucu + ayak bileği) rakibin baş merkezine uzaklığıyla ölçülür.
# Eşikler BAŞ YARIÇAPI cinsindendir; böylece kamera uzaklığından bağımsızdır.
#   uzaklık <= (1 + CONTACT_MARGIN_RADII) yarıçap   -> temas
#   uzaklık <= (1 + NEAR_MISS_MARGIN_RADII) yarıçap -> yakın geçiş (near miss)
# Pay, ayak genişliği ve iskelet noktası tahmin hatası içindir.
CONTACT_MARGIN_RADII = 0.5
NEAR_MISS_MARGIN_RADII = 1.5
# Not: İki boyutlu görüntüde temas ile temassız geçiş ayırt edilemeyebilir (bkz. YOL_HARITASI.md,
# Bulgu 3). Bu yüzden temas kararda sadece destekleyici bir kanıttır.
