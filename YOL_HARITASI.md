# Yol Haritası

Projenin geliştirme planı ve her adımın durumu. Her adım ayrı bir commit olarak GitHub'a gönderilir; commit açıklamalarında 52 maddelik inceleme listesinden hangi maddelerin kapandığı yazar. Doğruluk değerleri `python degerlendir.py` ile 20 orijinal klip üzerinde ölçülür (geçmiş: `output/reports/degerlendirme_gecmisi.csv`).

**Başlangıç doğruluğu:** 11/20 (eşikler test verisine göre ayarlanmıştı) · **Güncel dürüst doğruluk:** %55 (11/20, birini dışarıda bırakarak; `python egit.py`)

---

## Temel bulgular (yönü belirleyen)

1. **Gerçek darbe başı döndürür.** Real Hit kliplerinde burun noktası, baş merkezinden ~%35 daha fazla hareket ediyor (ortanca 140 px'e karşı 104 px); Ghost Hit kliplerinde ikisi aynı (~100 px). *(Adım 8)*
2. **Poz modeli yüksek tekmede kafaya uzanan bacağı yakalayamıyor.** Daha büyük poz modeli (YOLOv8m) de yakalamadı ve doğruluğu düşürdü (13 → 10). *(Adım 9)*
3. **İki boyutlu görüntüde temas, temassız geçişten ayırt edilemiyor.** Segmentasyon (siluet) bacağı yakalasa da, saldıranın silüetinin rakip başı kaplama oranı Ghost Hit kliplerinde daha yüksek çıktı (ortanca %36'ya karşı %10): Ghost Hit'te de tekme başın hemen yanından geçiyor. *(Adım 11 sonrası test)*

4. **Başın en yüksek ivmesi ayırt edici değil.** Gürültü giderildikten sonra bile en yüksek ivme Ghost Hit kliplerinde Real Hit'tekinden yüksek (ortanca 13.526'ya karşı 10.615 referans px/sn²). Eski rapordaki "gerçek darbe = keskin ivme tepesi" varsayımı veride doğrulanmıyor; ayırt edici olan, darbe anından sonraki savrulma ve dönüş. *(Adım 13)*

5. **Baş dönüşü en ayırt edici ölçüm; başın gövdeye göre savrulması ise değil.** 20 klipte AUC (rastgele bir Real Hit'in rastgele bir Ghost Hit'ten yüksek değer alma olasılığı; %50 = şans): baş dönüşü %69, burnun mutlak savrulması %67, başın gövdeye göre savrulması %46, en yüksek ivme %41. Ölçümler klibin tamamından alındığı için temas anına odaklanınca (Adım 15) değişebilir; 10'a 10 klipte belirsizlik payı büyüktür. *(Adım 14)*

6. **Dürüst doğruluk şans düzeyine yakın.** Karar modeli her klip (tüm kopyalarıyla) bir kez dışarıda tutularak değerlendirildiğinde: tüm ölçümlerle 11/20 (%55), sadece baş dönüşüyle 10/20, sadece burun savrulmasıyla 11/20. Önceki adımlardaki 11-13/20 sonuçları, eşikleri aynı veriye göre ayarlanmış kararlardan geliyordu. Mevcut ölçümler 20 klipte belirgin bir eğilim gösterse de (AUC %67-69), bu eğilim görülmemiş kliplere genellenecek kadar güçlü değil. En olası nedenler: veri azlığı (10'a 10 klip), otomatik temas anının belirsizliği ve iki boyutlu görüntünün sınırları. *(Adım 17-18)*

**Sonuç:** Kararın ana dayanağı **başın darbeye verdiği fiziksel tepki** (dönüş, gövdeye göre savrulma) olmalı; temas geometrisi sadece destekleyici kanıttır.

---

## Adımlar

| # | Adım | Durum | Kapatılan maddeler |
|---|---|---|---|
| | **1. Aşama – Temel kararlar ve düzen** | | |
| 1 | Ghost Hit tanımını sabitlemek | ✅ | 52, 30 (kısmen) |
| 2 | Proje klasörünü düzenlemek, tek giriş dosyası | ✅ | 45, 46 |
| 3 | Değerlendirme aracı (`degerlendir.py`) | ✅ | 37, 38 |
| | **2. Aşama – Algılama** | | |
| 4 | YOLO öncesi bulanıklaştırmayı kaldırmak | ✅ | 7, 10 |
| 5 | Videonun gerçek FPS değerini kullanmak | ✅ | 9 |
| 6 | Sporculara kalıcı kimlik (ByteTrack + renk oylaması) | ✅ | 11, 12, 13 |
| 7 | Nokta güveni + parlaklıktan bağımsız renk tespiti | ✅ | 14 (kısmen) |
| 8 | Kafa merkezi ve kafa yarıçapı | ✅ | 15, 14 |
| 9 | Ayak ucu tahmini + darbe alan sporcunun takibi | ✅ | 16 |
| 10 | Mesafeleri sporcu boyuna göre ölçeklemek | ✅ | 20, 8 |
| | **3. Aşama – Analiz** | | |
| 11 | Temas anını bulmak (kullanıcı karesi > yaklaşım > baş hareketi) | ✅ | 34 (kısmen) |
| 12 | Temas ölçümünü kafa yarıçapı ve ayak ucuyla yeniden yazmak *(küçültüldü: temas artık destekleyici kanıt)* | ✅ | 19, 21, 43 (kısmen) |
| 13 | Hareket verisini doğru yumuşatmak (Savitzky–Golay) | ✅ | 24 |
| 14 | **Başın gövdeye göre hareketi ve baş dönüşü** *(ana özellik)* | ✅ | 25 |
| 15 | Kinematik analizi temas anına odaklamak *(sadece an güvenilirse: kullanıcı karesi / gerçek yaklaşım)* | ✅ | 26, 27, 28, 29 (kısmen) |
| 16 | Takip modülünü sadeleştirmek *(eski tracker.py kaldırıldı)* | ✅ | 22, 23 |
| | **4. Aşama – Karar** | | |
| 17 | Karar motorunu yeni tanıma göre yazmak (temas var / yok / belirsiz × sensör) | ✅ | 30, 32, 34 |
| 18 | Elle yazılmış eşikler yerine yorumlanabilir küçük model + doğru değerlendirme *(17 ile birlikte)* | ✅ | 33, 35, 36 |
| 19 | Gerçek güven değeri ve otomatik gerekçe *(olasılık ve gerekçe 17-18'de eklendi; kalan: güvenin kalibrasyonu)* | | 31 (kısmen) |
| | **5. Aşama – Arayüz ve rapor** | | |
| 20 | Analizi tek geçişte yapmak | | 39, 40 |
| 21 | Video ile grafikleri eşzamanlamak | | 41 |
| 22 | Sensör durumu seçimi, temas anını görünür yapmak | | 44 |
| 23 | Gerçek ilerleme çubuğu | | 42 |
| 24 | PDF raporu yenilemek | | 43, 49 (kısmen) |
| | **6. Aşama – Kalite ve dokümantasyon** | | |
| 25 | Otomatik testler | | 48 |
| 26 | README'yi yeniden yazmak | | 47, 49, 51 |
| 27 | Sonuç raporu (doğruluk, karmaşıklık matrisi, hatalar) | | 50 |

## Denenip vazgeçilenler

| Deneme | Sonuç | Neden vazgeçildi |
|---|---|---|
| Düşük güvenli tüm noktaları atmak (Adım 7) | 13 → 10 | Yerine konacak bir şey olmadığı için kafa/ayak kayboluyordu |
| Hareket analizinde burun yerine kafa merkezi (Adım 8) | 12 → 10 | Baş dönüşü bilgisini siliyordu (Bulgu 1) |
| YOLOv8m (büyük) poz modeli | 13 → 10, 4 kat yavaş | Tekme atan bacağı yine yakalamadı |
| YOLOv8n-seg siluet ile temas | Ayrıştırmadı | Bulgu 3 |
| Ölçümleri otomatik bulunan temas anına odaklamak (Adım 15) | Baş dönüşü AUC %69 → %52 | Otomatik an kliplerin yarısında yanlış kareye düşüyor; odaklanma artık sadece güvenilir anda yapılıyor |
