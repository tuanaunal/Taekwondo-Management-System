# deneyler/

Bu klasör, projenin geliştirilmesi sırasında yazılan **deneme ve hata ayıklama betiklerini** içerir. Ana sistemin (arayüz, analiz, karar) çalışması için hiçbiri gerekli değildir; geliştirme geçmişini göstermek ve ileride başvurmak için saklanmaktadır.

Betikler proje kök klasöründen çalıştırılmalıdır, örneğin:

```bash
python deneyler/inspect_video.py
```

| Dosya grubu | Ne işe yarıyordu |
|---|---|
| `inspect_*.py`, `find_*_hsv.py`, `analyze_blue_hsv.py` | Tek bir videonun karelerini ve renk (HSV) değerlerini incelemek |
| `check_helmet_contact.py`, `find_actual_contact.py` | Temas anını ve kask–ayak mesafesini elle incelemek |
| `debug_*.py`, `scratch_debug.py` | Yanlış sınıflandırılan videoları ayıklamak |
| `test_*.py` | Takip ve renk atama yöntemlerinin eski denemeleri |
| `analyze_ke.py`, `optimize_ke.py`, `find_thresholds.py`, `train_tree.py` | Karar eşiklerini toplu sonuçlar üzerinden ayarlama denemeleri |

> Not: Eşik ayarlama betikleri, eşikleri test edilen verinin kendisi üzerinde ayarlar (ezberleme riski). Doğru değerlendirme yöntemi yol haritasının 18. adımında kurulacaktır.
