# Taekwondo Ghost Hit Detection System 🥋🤖

Bu proje, Taekwondo müsabakalarında yaşanan kask sensörü hatalarını (Ghost Hit - Hayalet Darbe) tespit etmek ve gerçek darbeleri (Real Hit) ayırt etmek amacıyla geliştirilmiş yapay zeka destekli, resmi bir **Karar Destek Sistemi (Decision Support System)**'dir.

## 🚀 Proje Hakkında

Müsabakalarda bazen sporcuların kaskları, bir tekme veya temas olmamasına rağmen ani hareketler veya savrulmalar sebebiyle darbe sensörünü tetiklemektedir. Bu durum hakemleri yanıltarak haksız puanlama (Ghost Hit) yaşanmasına neden olur.

Bu sistem, gelişmiş Görüntü İşleme (OpenCV) ve Obje Takibi algoritmalarını (YOLOv8 Pose & Segmentasyon) birleştirerek **"Çift Ayak ve Kask Takibi (Dual Foot Tracking)"** yapar. Gerçekleşen olayı sadece piksellerle değil, matematiksel ve kinematik temellere dayandırarak çözümler.

---

## 🧠 Mantıksal Mimari ve Beyin (Decision Engine)

Sistem ezbere dayalı bir yapı değil, **dinamik** bir karar mekanizmasıdır. Videodaki hareketleri kare kare işleyerek kendi kararını kendi verir:

1. **Çift Ayak Takibi (Dual Foot Tracking):** Sadece vuran ayağı değil, sporcuların **iki ayağını birden** tespit edip, kaska en yakın (tehlike potansiyeli yüksek) olan ayağı dinamik olarak filtreler.
2. **Öklid (Euclidean) Mesafe Analizi:** Ayak ile kask arasındaki mesafeyi sürekli ölçer. Hızlı tekmelerde (Bulanıklık/Motion Blur) ayak tespit edilemese bile "Son Bilinen Konum" (Last Known Position) üzerinden kinematik tahmin yürütür.
3. **Piksel Çakışması (Overlap):** Kask ve ayak bölgelerinin matematiksel olarak ne kadar iç içe girdiğini hesaplar.
4. **Kinematik Analiz:** Kafadaki savrulmanın ivmesini (px/s²) hesaplar. Sensörü tetikleyen şeyin gerçekten bir darbe mi yoksa sadece kafayı sertçe çevirmek mi olduğunu ivme profiliyle anlar.

Bu veriler harmanlanır ve sistem kesin kararını verir: **REAL HIT (Gerçek Darbe) veya GHOST HIT (Hayalet Darbe)**.

---

## ✨ Modern Arayüz ve Raporlama (GUI)

Proje sadece arka planda çalışan bir kod değil, aynı zamanda müsabaka sırasında rahatlıkla kullanılabilecek **Premium bir Arayüze** sahiptir:

- **Siber Tasarım (Dark Mode):** Rounded (yuvarlatılmış) hatlar, interaktif hover efektleri ve şık veri kutuları (GroupBox).
- **Gerçek Zamanlı Analiz Grafikleri:** O anki ivme hızını ve ayak-kask mesafesini saniye saniye çizen grafik ekranı.
- **Dinamik Video Oynatıcı:** İstediğin kareye saniye saniye gidebileceğin ve analiz bitince otomatik başa saran akıllı Player.
- **Resmi PDF Rapor Çıktısı (fpdf):** Analiz tamamlandığında "Rapor Kaydet" butonuna tıklandığında Masaüstüne resmi, A4 boyutunda, sayfa numaralı, kurumsal bir **"TAEKWONDO GHOST HIT - ANALİZ RAPORU"** PDF'i çıkartır.

---

## 🛠️ Kurulum ve Kullanım

### Gereksinimler
- Python 3.8+
- OpenCV, PyQt5, matplotlib, fpdf, numpy

### Kurulum
1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone https://github.com/tuanaunal/Taekwondo-Management-System.git
   ```
2. Gerekli Python kütüphanelerini kurun:
   ```bash
   pip install -r requirements.txt
   ```
   *(Eğer gereksinim dosyası yoksa: `pip install opencv-python PyQt5 matplotlib fpdf numpy`)*

### Çalıştırma
Sistemi modern arayüz ile başlatmak için terminalden şu komutu çalıştırmanız yeterlidir:
```bash
python main.py
```
Açılan pencerede **"Video Yükle"** diyerek `.mp4` formatındaki müsabaka kaydını seçip, **"Analizi Başlat"** butonuyla sistemi harekete geçirebilirsiniz.

---

## 🔒 Veri Gizliliği

KVKK ve yarışma mahremiyeti kuralları gereği, projede test amacıyla kullanılan asıl müsabaka videoları bu açık kaynak repository üzerinde paylaşılmamıştır.

**Geliştirici:** Tuana Ünal
