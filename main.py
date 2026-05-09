import cv2
import numpy as np
import os

# 1. Klasör yapısını tanımlıyoruz
base_path = "Standard_Dataset"
kategoriler = ["Ghost_Hit", "Real_Hit"]

# Işık dengeleme (CLAHE) ayarı
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# 2. Döngü: Önce kategorileri (Ghost/Real), sonra içindeki videoları geziyoruz
for kategori in kategoriler:
    klasor_yolu = os.path.join(base_path, kategori)
    videolar = [f for f in os.listdir(klasor_yolu) if f.endswith('.mp4')]
    
    for video_adi in videolar:
        video_tam_yol = os.path.join(klasor_yolu, video_adi)
        cap = cv2.VideoCapture(video_tam_yol)
        
        print(f"Şu an işlenen: {kategori} -> {video_adi}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ROI - Sadece dövüş alanına odaklan
            roi = frame[100:650, 100:1100] 

            # CLAHE - Işığı Dengele
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            final_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Mavi Kask Tespiti
            hsv = cv2.cvtColor(final_roi, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Ekranda hangi videoda olduğumuzu yazalım (Görsel takip için)
            cv2.putText(roi, f"{kategori}: {video_adi}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Analiz Penceresi', roi)
            cv2.imshow('Bilgisayarin Gozu (Maske)', blue_mask)

            # 'q' ile videoyu atlayabilir veya 'esc' ile tamamen kapatabilirsin
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'): # Mevcut videoyu geçer, sıradakine başlar
                break
            elif key == 27: # ESC tuşu programı tamamen kapatır
                cap.release()
                cv2.destroyAllWindows()
                exit()

        cap.release()

cv2.destroyAllWindows()
print("Tüm veri seti analizi tamamlandı!")