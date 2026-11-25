# Darbe Karesi İşaretleme

Amaç: Her klipte **darbenin (temas anının) hangi karede olduğunu** belirlemek. Sistem baş dönüşü ve savrulmayı bu anın hemen öncesi ve sonrasından ölçecek. Sahadaki gerçek kullanımda bu kareyi hakem verir (puanlama sistemi puanın anını kaydeder); bu işaretleme o senaryoyu test eder.

## Nasıl yapılır?

1. `kareler/` klasöründe her klip için bir resim var (örn. `RH_07.jpg`). Resimde klibin tüm kareleri soldan sağa, yukarıdan aşağıya sıralı; **kare numarası her karenin sol üst köşesinde sarı yazıyla** yazıyor. Ayrıntı için resmi yakınlaştırın.
2. `darbe_kareleri.csv` dosyasını açın (Excel veya Not Defteri) ve her satırın `darbe_karesi` sütununa kare numarasını yazın:
   - **Real Hit (RH) klipleri:** ayağın başa **ilk değdiği** kare.
   - **Ghost Hit (GH) klipleri:** tekmenin başa **en yakın geçtiği** kare (sensörün yanlış tetiklendiği an).
   - Emin değilseniz en yakın tahmini yazın ve `not` sütununa "emin değilim" yazın.
   - Klipte birden fazla tekme varsa, tartışmalı olanı (puanın verildiği/itiraz edilen) seçin ve `not` sütununa kısaca belirtin.
3. Dosyayı kaydedin. Örnek satır:

```
RH_07,24,
GH_03,18,emin değilim
```

Dosyayı doldurduktan sonra sistem ölçümleri bu karelere odaklayarak modeli yeniden değerlendirecek.
