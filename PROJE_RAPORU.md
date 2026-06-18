# Hayvan Hastalıkları Sınıflandırma — Proje Raporu

**Tarih:** 2026-06-13
**Donanım:** NVIDIA RTX 3060 Laptop GPU (Ampere, compute 8.6, 6 GB VRAM)
**Ortam:** PyTorch 2.11 + CUDA 12.8, Windows 11

---

## 1. Genel Bakış

Proje, hayvan hastalıklarını görüntüden sınıflandırmak için **hayvan-bazlı** (cat, dog, cattle, pig, poultry) derin öğrenme modelleri eğitir ve değerlendirir. Her hayvan kendi hastalık sınıfı alt kümesiyle ayrı ayrı eğitilir; böylece her model yalnızca ilgili hayvanın hastalıklarına odaklanır.

**Eğitilen 5 mimari (her hayvan için):**
| Model | Kaynak | Tip |
|---|---|---|
| ConvNeXtV2-Base | `facebook/convnextv2-base-22k-224` (HF) | CNN (modern) |
| EfficientNetV2-M | torchvision IMAGENET1K_V1 | CNN |
| CvT-21 | `microsoft/cvt-21` (HF) | Convolutional Transformer |
| ResNeXt101-32x8d | torchvision IMAGENET1K_V2 | CNN |
| ResNeSt101e | timm | CNN (split-attention) |

**Veri:** `./split/{train, val, test}` — 33 sınıf, hayvana göre filtrelenir.

---

## 2. Eğitim Stratejisi (`train_animal.py`)

- **Transfer learning + kademeli açma:**
  - **Phase 1 (Freeze):** İlk 3 epoch backbone donuk, sadece head eğitilir.
  - **Phase 2 (Warmup + Unfreeze):** Tüm katmanlar açılır, 3 epoch lineer warmup, sonra tam eğitim.
- **Katmanlı LR:** Head ve backbone farklı öğrenme oranı (backbone = head × 0.1).
- **Optimizasyon:** AdamW, weight decay 1e-4, gradient clipping (norm=1.0).
- **Dengesizlik önlemi:** Class-weighted CrossEntropy + label smoothing (0.1).
- **Scheduler:** ReduceLROnPlateau (val acc'a göre).
- **Early stopping:** 7 epoch sabır.
- **Checkpoint:** Her epoch `latest` + en iyi `best`. Resume desteği (`--resume latest`).
- **AMP:** Mixed precision; ConvNeXtV2 GRN için Windows/CUDA FP16 monkey-patch'i.

### Çıktı yapısı
```
checkpoints/run_animal_20260610_1217/
  ├─ cat/      ├─ dog/    ├─ cattle/   ├─ pig/    └─ poultry/
        └─ <Model>_best_v1.pth   (her hayvanda 5 model)
```

---

## 3. Bu Oturumda Yapılan Kod Değişiklikleri

### 3.1 Hız/Stabilite iyileştirmeleri — `train_animal.py` + `train_v2.py`
GPU'nun (Ampere) desteklediği donanım hızlandırmaları **eklendi, hiçbir şey silinmedi**:

| # | Değişiklik | Yer | Etki |
|---|---|---|---|
| 1 | **TF32** (`allow_tf32=True`) | `set_seed()` | +%10-15 hız |
| 2 | **bf16 autocast** (`AMP_DTYPE`) | train/val döngüleri | fp16'dan stabil |
| 3 | **channels_last** bellek formatı | model + input'lar | +%10-20 hız (CNN) |
| 4 | **DataLoader** workers 4→8, prefetch 2→4 | `main()` | GPU az bekler |

> **Not:** `num_workers=8` bir çalıştırmada segfault'a (worker çökmesi) yol açtı. Tekrar eğitimde 4'e çekilmesi önerilir. ConvNeXtV2 GRN patch'ine dokunulmadı.

### 3.2 Checkpoint konsolidasyonu
- ConvNeXt ayrı klasörde (`run_animal_20260612_0956`) eğitilmişti; her hayvanın `ConvNeXtV2-Base_best_v1.pth` dosyası ana klasöre (`run_animal_20260610_1217`) kopyalandı.
- Artık her hayvanda 5 model yan yana. Ayrı ConvNeXt klasörü silindi (tek temiz klasör kaldı).

### 3.3 Yeni test/analiz scripti — `test/test_animal_final.py`
`test_finalv5.py` temel alındı, **yeni hayvan-bazlı yapıya uyarlandı**:
- Model kurucuları doğrudan `train_animal.py`'den import → mimari sapması yok.
- `AnimalFilteredImageFolder` ile her hayvan kendi sınıf alt kümesinde değerlendirilir.
- Her hayvan × model × split (train/val/test) için üretilenler:
  - **Confusion Matrix, ROC eğrileri, Precision-Recall eğrileri**
  - **Sınıf-bazlı Precision/Recall/F1 bar grafiği**
  - **`report.json`** (tüm metrikler)
- `test_full_vision.py`'den eklenen ek metrikler: **MCC, Cohen's Kappa, Balanced Accuracy** (+ terminalde renkli ASCII bar).
- Sonuçlar: `final_analiz_raporu_animal/<hayvan>/<model>/<split>/...` + toplu `ozet_tum_sonuclar.csv`.

---

## 4. Test Sonuçları (TEST split, Macro F1'e göre)

> Metrikler: **Acc** = doğruluk, **F1** = macro F1, **AUC** = macro AUC, **MCC** = Matthews, **Kappa** = Cohen's Kappa.

### 🏆 Her hayvanın en iyi modeli
| Hayvan | En İyi Model | Acc | Macro F1 | Macro AUC | MCC |
|---|---|---|---|---|---|
| **cat** | ResNeSt101e | 0.865 | **0.854** | 0.979 | 0.842 |
| **dog** | ConvNeXtV2-Base | 0.972 | **0.973** | 0.997 | 0.968 |
| **cattle** | ConvNeXtV2-Base | 0.905 | **0.896** | 0.991 | 0.877 |
| **pig** | ConvNeXtV2-Base | 0.909 | **0.908** | 0.986 | 0.891 |
| **poultry** | ConvNeXtV2-Base | 0.971 | **0.968** | 1.000 | 0.965 |

**Özet:** ConvNeXtV2-Base 5 hayvanın 4'ünde en iyi; cat'te ResNeSt101e öne geçti.

### Detaylı tablo (TEST split, her hayvan × model — Macro F1)
| Model | cat | dog | cattle | pig | poultry |
|---|---|---|---|---|---|
| **ConvNeXtV2-Base** | 0.825 | **0.973** | **0.896** | **0.908** | **0.968** |
| **ResNeSt101e** | **0.854** | 0.914 | 0.874 | 0.877 | 0.945 |
| **CvT-21** | 0.815 | 0.886 | 0.894 | 0.880 | 0.912 |
| **EfficientNetV2-M** | 0.707 | 0.827 | 0.858 | 0.890 | 0.920 |
| **ResNeXt101-32x8d** | 0.678 | 0.847 | 0.863 | 0.856 | 0.882 |

### Gözlemler
- **En güçlü hayvan:** dog ve poultry (F1 ≈ 0.97) — sınıflar daha ayırt edilebilir.
- **En zorlu hayvan:** cat (en iyi F1 0.854) — kedi hastalıkları görsel olarak benzer.
- **En zayıf model:** ResNeXt101 ve EfficientNetV2-M genelde geride; özellikle cat'te ResNeXt düşük (0.678).
- **Genelleme:** train→test arası düşüş makul (overfitting kontrol altında); ConvNeXtV2 dog'da train 0.990 → test 0.973 ile çok iyi genelliyor.

---

## 5. Üretilen Çıktılar

| Yol | İçerik |
|---|---|
| `checkpoints/run_animal_20260610_1217/<hayvan>/` | 5 modelin best ağırlıkları |
| `final_analiz_raporu_animal/ozet_tum_sonuclar.csv` | Tüm hayvan/model/split metrik tablosu |
| `final_analiz_raporu_animal/<hayvan>/<model>/<split>/` | Confusion Matrix, ROC, PR, F1 bar, report.json |
| `test/test_animal_final.py` | Kapsamlı test/analiz scripti |
| `train_animal.py`, `train_v2.py` | Hızlandırma güncellemeli eğitim scriptleri |

---

## 6. Öneriler / Açık Konular

1. **Segfault önlemi:** Tekrar eğitimde `num_workers`'ı 4'e çek (8 worker laptop'ta worker çökmesi yaptı).
2. **VRAM:** 6 GB için ConvNeXt ağır; sorun olursa `--batch_size 16-24`.
3. **cat performansı:** En zayıf alan; ek veri artırma / daha fazla epoch / ResNeSt'e odaklanma düşünülebilir.
4. **Ensemble:** Üretimde her hayvan için en iyi 2-3 modeli (ConvNeXtV2 + ResNeSt + CvT) birleştirmek doğruluğu artırabilir.
5. **Dağıtım:** Her hayvan için en iyi model seçilip tek bir "yönlendirici + uzman model" mimarisi kurulabilir.
