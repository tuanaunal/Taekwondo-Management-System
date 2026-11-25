# Taekwondo'daki Elektronik Puanlama Hatalarının Bilgisayarlı Görü ile Tespiti: Hakem Karar Destek Sistemi 🥋

Taekwondo kurallarına göre rakibin kafasına yapılan geçerli her vuruş puan değeri taşır; ancak günümüz müsabakalarında bu puanı hakemler değil, elektronik bir sistem verir. Puan yalnızca sporcunun ayak üstü koruyucusundaki sensör ile rakibin kaskındaki sensör eşleştiğinde oluşur. Bu yüzden kafaya isabet eden bir vuruş sensöre denk gelmediğinde puan almayabilir; sensörler bazen de temas olmadan tetiklenerek hatalı puan verebilir ("Ghost Hit"). Tartışmalı kararlar hâlâ hakemlerin gözle yaptığı incelemeye bağlıdır.

Bu proje, sensörden bağımsız olarak yalnızca müsabaka görüntüsünü analiz edip **kafaya gerçek bir temas olup olmadığını** değerlendiren bir **hakem karar destek sistemidir**. Sistem hakemin yerini almak için değil, ona saniyeler içinde ikinci bir görüş sunmak için tasarlanmıştır.

> **Önemli:** Sistemin, hiç görmediği kliplerde ölçülen dürüst doğruluğu şu an **%55**'tir (20 klip, birini dışarıda bırakarak). Ayrıntılar ve nedenleri için [Sonuçlar ve sınırlar](#-sonuçlar-ve-sınırlar) bölümüne ve [YOL_HARITASI.md](YOL_HARITASI.md) dosyasına bakın.

---

## 📌 Problem tanımı

Kafa vuruşlarında iki tür puanlama hatası oluşur:

1. **Ghost Hit (hayalet darbe):** Sensör puan verir, ama kafaya gerçek bir temas yoktur.
2. **Kaçırılan vuruş:** Kafaya gerçek bir temas vardır, ama vuruş sensöre denk gelmediği için puan çıkmaz.

Sistem görüntüden **temas var / temas yok / belirsiz** kararı verir. Hakem sensörün puan verip vermediğini girerse karar şu tabloya göre birleşir:

| | Görüntü: temas var | Görüntü: temas yok |
|---|---|---|
| **Sensör puan verdi** | ✅ Puan geçerli | 🔴 Ghost Hit (puan hatalı) |
| **Sensör puan vermedi** | 🟠 Kaçırılan vuruş (puan verilmeli) | ✅ Puan yok, doğru karar |

Görüntü karar vermeye yetmiyorsa (ör. darbe alan sporcunun başı karelerin yarısından azında görülebiliyorsa) sonuç **belirsiz**dir.

---

## 🧠 Sistem nasıl çalışır?

```
Video ─► Kişi ve iskelet tespiti ─► Kırmızı/mavi sporcu seçimi ─► Baş ve ayak konumları
            (YOLOv8-Pose,             (kalıcı kimlik + göğüs        (baş merkezi ve yarıçapı,
             ByteTrack takip)          koruyucusu renk oylaması)      ayak ucu, gövde merkezi)
                                                                            │
     Karar ◄── Yorumlanabilir model ◄── Baş tepkisi + temas ◄── Temas anı ◄─┘
  (+ sensör     (lojistik regresyon,      (baş dönüşü, burun      (hakemin işaretlediği
   tablosu)      kalibre güven)            savrulması, ayak-baş    kare veya otomatik)
                                           mesafesi)
```

| Aşama | Ne yapılıyor | Dosya |
|---|---|---|
| Ön işleme | Işık dengeleme (CLAHE) | `src/preprocessor.py` |
| Algılama | YOLOv8-Pose ile kişiler ve iskelet noktaları; ByteTrack ile her kişiye video boyunca kalıcı kimlik | `src/segmentor.py` |
| Sporcu seçimi | Video tamamen tarandıktan sonra kırmızı ve mavi sporcu, *görünürlük × göğüs koruyucusu renk tutarlılığı* ile bir kez seçilir. Hakem ve arka plandaki kişiler elenir. Sporcunun kimliği darbe anında koparsa en yakın kişiyle sürdürülür | `src/segmentor.py` |
| Ölçek | Tüm mesafeler sporcunun baş yarıçapına göre ölçeklenir (kamera uzaklığından ve çözünürlükten bağımsız) | `src/segmentor.py` |
| Temas anı | Hakemin işaretlediği kare(ler); yoksa ayağın başa en çok yaklaştığı ya da başın en hızlı hareket ettiği kare | `src/contact_moment.py` |
| Baş tepkisi | **Baş dönüşü** (burnun baş merkezine göre kayması) ve burun savrulması; Savitzky–Golay ile yumuşatılmış hareket | `src/head_response.py`, `src/kinematics.py` |
| Temas | Ayak noktalarının rakibin baş merkezine uzaklığı (baş yarıçapı cinsinden) | `src/contact_analyzer.py` |
| Karar | Küçük bir lojistik regresyon "temas var" olasılığını hesaplar; gösterilen güven, benzer kararların dürüst ölçümdeki gerçek doğruluk oranıdır | `src/decision_engine.py` |
| Arayüz ve rapor | Video oynatıcı ve işaretler, videoyla eşzamanlı grafikler/tablo, hakem girdileri, tek sayfalık PDF rapor | `gui/`, `src/pdf_report.py` |

---

## 🛠️ Kurulum

Python 3.10 veya üzeri gerekir.

```bash
git clone https://github.com/tuanaunal/Taekwondo-Management-System.git
cd Taekwondo-Management-System
pip install -r requirements.txt
```

YOLOv8-Pose model dosyası (`yolov8n-pose.pt`) repoda bulunur. Windows'ta program `main.py` ile başlatılmalıdır (PyTorch'un PyQt5'ten önce yüklenmesi gerekir).

---

## ▶️ Kullanım

### Arayüz

```bash
python main.py
```

1. **Video Dosyası Aç** ile bir müsabaka kesiti seçin.
2. İsteğe bağlı **hakem girdileri**:
   - *Sensör puan verdi mi?* — Bilinmiyor / Puan verdi / Puan vermedi. Analizden sonra değiştirilirse karar anında güncellenir.
   - *Temas anı* — videoyu durdurup **Bu kareyi işaretle**. Birden fazla an işaretlenebilir; işaret yoksa an otomatik bulunur.
3. **Analizi Başlat**. İlerleme çubuğu işlenen kareleri gösterir.
4. Sonuç panelinde karar, güven ve gerekçe görünür. Grafikler ve kare tablosu videoyla birlikte ilerler; tablodan bir satıra tıklanınca video o kareye gider. **Overlay** düğmesi baş ve ayak işaretlerini gösterir.
5. **Rapor Kaydet** ile tek sayfalık PDF rapor alınır.

### Komut satırı

```bash
python main.py Dataset/Real_Hit/RH_07.mp4
```

Python içinden sensör durumu ve temas anı da verilebilir:

```python
from main import analyze_video
sonuc = analyze_video("video.mp4", sensor_scored=True, contact_ranges=[(23, 27)])
print(sonuc["decision_result"]["label_tr"], sonuc["decision_result"]["confidence"])
```

### Değerlendirme, model eğitimi ve testler

| Komut | Ne yapar |
|---|---|
| `python degerlendir.py` | 20 orijinal klipte doğruluğu ölçer; `--tumu` ile 120 videonun tamamında, görüntü koşullarına göre dayanıklılıkla birlikte. Sonuçlar `output/reports/degerlendirme_gecmisi.csv` dosyasına eklenir. *Not: model bu videolarla eğitildiği için buradaki doğruluk iyimserdir; araç dürüst doğruluğu ayrıca yazar.* |
| `python egit.py` | Ölçümleri çıkarır (~10-15 dk), modeli **birini dışarıda bırakarak** değerlendirir, güven kalibrasyonunu hesaplar ve `model/karar_modeli.json` dosyasına kaydeder. `--onbellek` önceki ölçümleri kullanır. |
| `python -m pytest` | 25 otomatik test (~4 sn; YOLO ve video gerektirmez). |

---

## 📊 Sonuçlar ve sınırlar

Veri seti: gerçek müsabaka kayıtlarından ayıklanıp etiketlenen **10 Real Hit ve 10 Ghost Hit klibi**; her biri görüntü işleme teknikleriyle 5 farklı koşulda çoğaltılmıştır (aynalama, parlaklık, karanlık, gürültü, bulanıklık; toplam 120 video).

**Dürüst doğruluk** (her klip tüm kopyalarıyla bir kez dışarıda tutularak):

| Ölçümler | 20 orijinal klip |
|---|---|
| Tüm ölçümler (kullanılan model) | **11/20 (%55)** |
| Sadece baş dönüşü | 10/20 |
| Sadece burun savrulması | 11/20 |

**Temel bulgular** (ayrıntılar: [YOL_HARITASI.md](YOL_HARITASI.md)):

1. **Gerçek darbe başı döndürür:** Real Hit kliplerinde burun, baş merkezinden ~%35 fazla hareket ediyor; Ghost Hit'te fark yok.
2. **İskelet modeli yüksek tekmede kafaya uzanan bacağı yakalayamıyor;** daha büyük model de çözmedi.
3. **İki boyutlu görüntüde temas, temassız geçişten ayırt edilemiyor:** Ghost Hit'te de tekme başın hemen yanından geçiyor.
4. **Başın en yüksek ivmesi ayırt edici değil.**
5. **Baş dönüşü en ayırt edici ölçüm** (20 klipte AUC %69), ancak bu eğilim görülmemiş kliplere genellenmiyor.
6. **Darbe anı hakem tarafından işaretlense de** ayırt edicilik artmıyor; temas anında iskelet noktaları en güvenilmez halinde.

**Sınırlar:** veri azlığı (10'a 10 klip), tek kameralı iki boyutlu yayın görüntüsü, temas anında üst üste binen sporcular. Sistem bu yüzden kararını her zaman **kalibre bir güven değeriyle** (şu an çoğunlukla "düşük") verir ve nihai kararı hakeme bırakır. Daha fazla etiketli veri, sistemin altyapısı değişmeden `python egit.py` ile modele eklenebilir.

---

## 📁 Proje yapısı

```
main.py              Giriş noktası: arayüz veya komut satırı analizi
degerlendir.py       Doğruluk ölçümü
egit.py              Karar modelinin eğitimi ve dürüst değerlendirmesi
augment.py           Veri çoğaltma (görüntü işleme)
src/                 Analiz modülleri (algılama, temas anı, baş tepkisi, karar, PDF rapor)
gui/                 PyQt5 arayüzü
model/               Eğitilmiş karar modeli (karar_modeli.json)
tests/               Otomatik testler
etiketleme/          Darbe karesi işaretleri ve işaretleme talimatı
deneyler/            Geliştirme sırasındaki deneme betikleri (arşiv)
YOL_HARITASI.md      Geliştirme adımları, bulgular, denenip vazgeçilenler
```

---

## 🔒 Veri gizliliği

KVKK ve yarışma mahremiyeti kuralları gereği, test için kullanılan müsabaka videoları bu açık kaynak depoda paylaşılmamıştır.

> Not: Depo adı ("Taekwondo-Management-System") projenin ilk halinden kalmıştır; proje bir yönetim sistemi değil, bir hakem karar destek sistemidir.

**Geliştirici:** Tuana Ünal
