import cv2
import os

ana_dataset = "Dataset"
cikti_klasoru = "Standard_Dataset"
kategoriler = ["Ghost_Hit", "Real_Hit"]

if not os.path.exists(cikti_klasoru):
    os.makedirs(cikti_klasoru)

for kat in kategoriler:
    eski_dizin = os.path.join(ana_dataset, kat)
    yeni_dizin = os.path.join(cikti_klasoru, kat)
    if not os.path.exists(yeni_dizin): os.makedirs(yeni_dizin)
    
    for vid in os.listdir(eski_dizin):
        if vid.lower().endswith(".mp4"):
            cap = cv2.VideoCapture(os.path.join(eski_dizin, vid))
            # 720p ve 30 FPS ayarları
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(yeni_dizin, vid), fourcc, 30.0, (1280, 720))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                resized = cv2.resize(frame, (1280, 720))
                out.write(resized)
            
            cap.release()
            out.release()
            print(f"{vid} başarıyla yazıldı.")

print("İşlem TAMAM!")