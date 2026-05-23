import cv2
import numpy as np
import os
import sys
import io

# Windows konsol encoding duzeltmesi
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Proje kok dizin yollari
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
GHOST_HIT_DIR = os.path.join(DATASET_DIR, "Ghost_Hit")
REAL_HIT_DIR = os.path.join(DATASET_DIR, "Real_Hit")

def augment_video(video_path, output_dir):
    filename = os.path.basename(video_path)
    base_name, ext = os.path.splitext(filename)
    
    # 1. Kaynak videoyu ac
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Hata: Video acilamadi: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0
        
    # Uretecegimiz 5 farkli veri artirma yontemi
    aug_types = ["flipped", "bright", "dark", "noisy", "blur"]
    writers = {}
    
    # Windows'ta yuksek uyumluluk saglayan mp4v codec'i
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for aug in aug_types:
        out_path = os.path.join(output_dir, f"{base_name}_{aug}{ext}")
        writers[aug] = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
    print(f"  [Isleniyor] {filename} ({width}x{height}, {fps:.0f} FPS, {frame_count} kare)...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Yatay Aynalama (Flipped)
        frame_flipped = cv2.flip(frame, 1)
        writers["flipped"].write(frame_flipped)
        
        # 2. Parlaklik Artirma (Bright)
        frame_bright = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        writers["bright"].write(frame_bright)
        
        # 3. Parlaklik Azaltma (Dark)
        frame_dark = cv2.convertScaleAbs(frame, alpha=0.7, beta=-15)
        writers["dark"].write(frame_dark)
        
        # 4. Gürültü Ekleme (Noisy)
        noise = np.random.normal(0, 12, frame.shape).astype(np.float32)
        frame_noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writers["noisy"].write(frame_noisy)
        
        # 5. Bulaniklastirma (Blur)
        frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
        writers["blur"].write(frame_blur)
        
        frame_idx += 1
        
    cap.release()
    for aug in aug_types:
        writers[aug].release()
        
    print(f"  [Tamamlandi] {filename} için 5 yeni video varyasyonu olusturuldu.")

def main():
    print("=" * 60)
    print("  TAEKWONDO VERI SETI COGALTMA (DATA AUGMENTATION) ARACI")
    print("=" * 60)
    
    categories = [
        ("Ghost_Hit", GHOST_HIT_DIR),
        ("Real_Hit", REAL_HIT_DIR)
    ]
    
    total_processed = 0
    for cat_name, cat_dir in categories:
        if not os.path.exists(cat_dir):
            print(f"Hata: Dizin bulunamadi: {cat_dir}")
            continue
            
        # Sadece orijinal videolari bul (artirilmis olanlar isleme alinmaz)
        all_files = sorted([f for f in os.listdir(cat_dir) if f.endswith(".mp4")])
        original_files = []
        for f in all_files:
            if not any(tag in f for tag in ["_flipped", "_bright", "_dark", "_noisy", "_blur"]):
                original_files.append(f)
                
        print(f"\nKategori: {cat_name} (Toplam {len(original_files)} orijinal video bulundu)")
        print("-" * 50)
        
        for f in original_files:
            video_path = os.path.join(cat_dir, f)
            augment_video(video_path, cat_dir)
            total_processed += 1
            
    print("\n" + "=" * 60)
    print(f"Veri Artirma Islemi Tamamlandi! {total_processed} video islendi.")
    print(f"Toplam 100 yeni video sentezlenerek veri setine eklendi.")
    print("=" * 60)

if __name__ == "__main__":
    main()
