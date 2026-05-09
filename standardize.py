import os
from moviepy import VideoFileClip

# 1. Klasör yollarını belirliyoruz
ana_dataset = "Dataset"
cikti_klasoru = "Standard_Dataset"

# Çıktı klasörü yoksa oluşturuyoruz
if not os.path.exists(cikti_klasoru):
    os.makedirs(cikti_klasoru)

# 2. Her iki kategoriyi de (Ghost ve Real) aynı anda işlemek için liste
kategoriler = ["Ghost_Hit", "Real_Hit"]

for kategori in kategoriler:
    # Alt klasörleri (Ghost_Hit ve Real_Hit) yeni yerde de oluşturuyoruz
    yeni_alt_klasor = os.path.join(cikti_klasoru, kategori)
    if not os.path.exists(yeni_alt_klasor):
        os.makedirs(yeni_alt_klasor)

    # Orijinal klasördeki videoları tarıyoruz
    eski_alt_klasor = os.path.join(ana_dataset, kategori)
    videolar = [f for f in os.listdir(eski_alt_klasor) if f.lower().endswith('.mp4')]

    print(f"\n--- {kategori} kategorisi işleniyor ---")

    for video_adi in videolar:
        eski_yol = os.path.join(eski_alt_klasor, video_adi)
        yeni_yol = os.path.join(yeni_alt_klasor, video_adi)

        print(f"İşleniyor: {video_adi}...")

        try:
            # 3. Standardizasyon İşlemleri
            clip = VideoFileClip(eski_yol)

            # Çözünürlüğü 720p (1280x720) yapıyoruz 
            # FPS'i 30'a sabitliyoruz (Matematiksel tutarlılık için) 
            new_clip = clip.resize(height=720).set_fps(30)

            # 4. Kaydetme (Sesi kapatıyoruz çünkü projemiz görüntü tabanlı)
            new_clip.write_videofile(yeni_yol, codec="mpeg4", audio=False)
            
            clip.close()
            new_clip.close()
        except Exception as e:
            print(f"Hata oluştu ({video_adi}): {e}")

print("\n!!! Tüm videolar başarıyla standardize edildi !!!")