# Taekwondo'daki Elektronik Puanlama Hatalarının Bilgisayarlı Görü ile Tespiti: Hakem Karar Destek Sistemi 🥋

## 🚀 Proje Hakkında

Taekwondo kurallarına göre rakibin kafasına yapılan geçerli her vuruş puan değeri taşır; ancak günümüz müsabakalarında bu puanı hakemler değil, elektronik bir sistem verir. Kafaya gerçekten isabet eden bir vuruş sensöre tam denk gelmediğinde puana dönüşmeyebilir; sensörler ise bazen gerçek bir temas olmadan tetiklenerek hatalı puan verebilir ("Ghost Hit" – hayalet darbe). Bu iki durum nedeniyle tartışmalı kararlar hâlâ hakemlerin gözle yaptığı incelemeye bağlıdır.

Bu proje, sensörden bağımsız olarak yalnızca müsabaka görüntüsünü analiz edip kafaya gerçek bir temas olup olmadığını belirleyerek her iki soruna da çözüm getirmeyi amaçlar. Sistem, YOLOv8-Pose derin öğrenme modeliyle sporcuların iskelet noktalarını tespit eder; ayak ile baş arasındaki mesafeyi ve baştaki ani savrulmayı analiz ederek kararını grafikler ve PDF raporla gerekçelendirir. Hakemin yerini almak için değil, ona saniyeler içinde ikinci bir göz olmak için tasarlanmıştır.

---

## 📌 Problem Tanımı

Taekwondo'da kafa vuruşlarının puanını elektronik sistem verir: puan, sporcunun ayak üstü koruyucusundaki sensör ile rakibin kaskındaki sensör eşleştiğinde oluşur. Kurallara göre ise kafaya isabet eden her geçerli vuruş puan değeri taşır. Bu fark iki tür hataya yol açar:

1. **Ghost Hit (hayalet darbe):** Sensör puan verir, ama kafaya gerçek bir temas yoktur. Sporcu hak etmediği bir puan kazanır.
2. **Kaçırılan vuruş:** Kafaya gerçek bir temas vardır, ama vuruş sensöre denk gelmediği için puan çıkmaz.

**Sistemin cevapladığı soru:** Sensörden bağımsız olarak, yalnızca müsabaka görüntüsüne bakarak *"Tartışmalı anda kafaya gerçek bir temas oldu mu?"*

Sistemin kararı (temas var / temas yok / belirsiz), sensörün o anda puan verip vermediği bilgisiyle birleştirilerek nihai sonuç elde edilir:

| | Sistem: temas var | Sistem: temas yok |
|---|---|---|
| **Sensör puan verdi** | ✅ Puan geçerli | 🔴 Ghost Hit – puan hatalı |
| **Sensör puan vermedi** | 🟠 Kaçırılan vuruş – puan verilmeli | ✅ Puan yok, karar doğru |

Görüntü karar vermeye yetmiyorsa (ör. sporcular üst üste bindiğinde) sistem **"belirsiz"** sonucunu verir. Veri setindeki etiketler de bu soruya göredir: `Real_Hit` = kafaya gerçek temas var, `Ghost_Hit` = kafaya gerçek temas yok.

> Not: Bu tanım projenin hedef yapısıdır. Karar motoru bu tabloya göre yol haritasının 17. adımında yeniden yazılacaktır.

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
