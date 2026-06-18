"""
=========================================================================
 HAYVAN-BAZLI KAPSAMLI TEST / ANALİZ RAPORU
=========================================================================
 test_finalv5.py temel alınarak yeni (hayvan-bazlı) eğitime uyarlandı.

 - Modeller train_animal.py'deki kurucularla birebir aynı şekilde inşa edilir
   (ConvNeXtV2, EfficientNetV2-M, CvT-21, ResNeXt101, ResNeSt101e).
 - Her hayvan kendi filtrelenmiş sınıf alt kümesiyle değerlendirilir
   (AnimalFilteredImageFolder — eğitimdeki ile aynı mantık).
 - Her model x her split (train/val/test) için:
     Confusion Matrix, ROC, PR eğrileri, sınıf-bazlı F1/Precision/Recall barları,
     JSON rapor (+ macro AUC).
 - Sonunda hayvan x model karşılaştırma özeti (CSV + konsol tablosu).

 Çalıştırma (proje kökünden):
     python test/test_animal_final.py
=========================================================================
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # başsız (GUI'siz) ortamda da çalışsın
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize

# --- Proje kökünü path'e ekle (test/ alt klasöründen çalışsa bile) ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# train_animal.py ile BİREBİR aynı model kurucuları ve hayvan filtresi
from train_animal import (
    build_convnextv2, build_efficientnetv2, build_cvt, build_resnext, build_resnest,
    AnimalFilteredImageFolder,
)

# =========================================================
# AYARLAR
# =========================================================
DATA_DIR       = os.path.join(ROOT, "split")
CKPT_DIR       = os.path.join(ROOT, "checkpoints", "run_animal_20260610_1217")
SAVE_DIR_ROOT  = os.path.join(ROOT, "final_analiz_raporu_animal")
BATCH_SIZE     = 16
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANIMALS = ["cat", "dog", "cattle", "pig", "poultry"]

# Model anahtarı -> (dosya adı kökü, kurucu fonksiyon)
MODEL_REGISTRY = {
    "ConvNeXtV2-Base":  build_convnextv2,
    "EfficientNetV2-M": build_efficientnetv2,
    "CvT-21":           build_cvt,
    "ResNeXt101-32x8d": build_resnext,
    "ResNeSt101e":      build_resnest,
}

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Eğitimde val/valid fallback gibi, testte de esnek olalım
SPLIT_CANDIDATES = ["train", "val", "valid", "test"]

os.makedirs(SAVE_DIR_ROOT, exist_ok=True)


# =========================================================
# TERMINAL ÇIKTI YARDIMCISI
# =========================================================
def print_ascii_bar(label, score, color_code="\033[92m"):
    """Terminalde metriği çubuk grafik olarak basar."""
    bar_len = int(max(0.0, min(1.0, score)) * 20)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    reset = "\033[0m"
    print(f"      {label.ljust(15)}: {color_code}{bar}{reset} {score:.4f}")


# =========================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =========================================================
def plot_detailed_metrics(all_labels, all_preds, all_probs, class_names, out_dir, tag):
    """Confusion Matrix, ROC ve PR eğrilerini çizer."""
    n = len(class_names)

    # 1) Confusion Matrix
    plt.figure(figsize=(max(8, n), max(6, n * 0.8)))
    cm = confusion_matrix(all_labels, all_preds, labels=range(n))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{tag} - Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıf'); plt.xlabel('Tahmin Edilen')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_confusion_matrix.png"))
    plt.close()

    # Tek sınıflı durumda ROC/PR anlamsız -> atla
    if n < 2:
        return

    y_bin = label_binarize(all_labels, classes=range(n))

    # 2) ROC
    plt.figure(figsize=(10, 8))
    for i in range(n):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f'{tag} - ROC Eğrileri')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_roc_curves.png"))
    plt.close()

    # 3) Precision-Recall
    plt.figure(figsize=(10, 8))
    for i in range(n):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {ap:0.2f})')
    plt.title(f'{tag} - Precision-Recall Eğrileri')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc="lower left", fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_pr_curves.png"))
    plt.close()


def plot_f1_precision_recall_bars(report_dict, class_names, out_dir, tag):
    """Sınıf-bazlı Precision/Recall/F1 çubuk grafiği."""
    metrics = ['precision', 'recall', 'f1-score']
    present = [c for c in class_names if c in report_dict]
    data = {m: [report_dict[c][m] for c in present] for m in metrics}

    x = np.arange(len(present)); width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.2), 6))
    ax.bar(x - width, data['precision'], width, label='Precision', color='#1f77b4')
    ax.bar(x,         data['recall'],    width, label='Recall',    color='#ff7f0e')
    ax.bar(x + width, data['f1-score'],  width, label='F1-Score',  color='#2ca02c')
    ax.set_ylabel('Skorlar')
    ax.set_title(f'{tag} - Sınıf Bazlı Precision, Recall ve F1')
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=90, ha="center")
    ax.set_ylim(0, 1.1); ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_class_metrics_bar.png"))
    plt.close()


# =========================================================
# DEĞERLENDİRME
# =========================================================
@torch.no_grad()
def run_inference(model, loader):
    all_preds, all_labels, all_probs = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_one(animal, model_name, build_fn, loaders, class_names):
    ckpt_path = os.path.join(CKPT_DIR, animal, f"{model_name}_best_v1.pth")
    if not os.path.exists(ckpt_path):
        print(f"   ⏭️  {model_name}: checkpoint yok ({ckpt_path}) — atlanıyor.")
        return []

    num_classes = len(class_names)
    print(f"\n🚀 [{animal.upper()}] {model_name} analizi...")
    try:
        model = build_fn(num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"   ❌ {model_name} yüklenemedi: {e}")
        return []

    out_base = os.path.join(SAVE_DIR_ROOT, animal, model_name)
    rows = []
    for split_name, loader in loaders.items():
        split_dir = os.path.join(out_base, split_name)
        os.makedirs(split_dir, exist_ok=True)
        tag = f"{animal}_{model_name}_{split_name}"

        labels, preds, probs = run_inference(model, loader)
        plot_detailed_metrics(labels, preds, probs, class_names, split_dir, tag)

        report = classification_report(
            labels, preds, labels=range(num_classes),
            target_names=class_names, output_dict=True, zero_division=0)

        try:
            macro_auc = roc_auc_score(
                label_binarize(labels, classes=range(num_classes)),
                probs, multi_class='ovr', average='macro')
        except Exception:
            macro_auc = 0.0

        # Dengesiz veri için en güvenilir ek metrikler (full_vision'dan)
        mcc          = matthews_corrcoef(labels, preds)
        kappa        = cohen_kappa_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)

        report['macro_auc']         = macro_auc
        report['mcc']               = mcc
        report['cohen_kappa']       = kappa
        report['balanced_accuracy'] = balanced_acc

        with open(os.path.join(split_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        plot_f1_precision_recall_bars(report, class_names, split_dir, tag)

        macro_f1 = report['macro avg']['f1-score']
        acc      = report['accuracy']
        print(f"   ✅ {split_name.upper()} sonuçları:")
        print_ascii_bar("Accuracy",     acc,          "\033[94m")
        print_ascii_bar("Balanced Acc", balanced_acc, "\033[96m")
        print_ascii_bar("Macro F1",     macro_f1,     "\033[92m")
        print_ascii_bar("Macro AUC",    macro_auc,    "\033[92m")
        print_ascii_bar("MCC",          mcc,          "\033[93m")
        print_ascii_bar("Cohen Kappa",  kappa,        "\033[95m")
        rows.append({
            "animal": animal, "model": model_name, "split": split_name,
            "accuracy": acc, "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1, "macro_auc": macro_auc,
            "mcc": mcc, "cohen_kappa": kappa,
        })

    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return rows


def build_loaders_for_animal(animal):
    """Hayvana özel filtrelenmiş train/val/test loader'larını kurar."""
    loaders, class_names = {}, None
    for split in SPLIT_CANDIDATES:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        try:
            ds = AnimalFilteredImageFolder(split_dir, animal, transform=eval_transforms)
        except Exception as e:
            print(f"   [UYARI] {animal}/{split} yüklenemedi: {e}")
            continue
        if len(ds.classes) == 0:
            continue
        # 'val' ve 'valid' ikisi de varsa tekrarı önle
        key = 'val' if split in ('val', 'valid') else split
        loaders[key] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
        if class_names is None:
            class_names = ds.classes
    return loaders, class_names


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Cihaz: {DEVICE}")
    print(f"Veri:  {DATA_DIR}")
    print(f"Ckpt:  {CKPT_DIR}")
    print(f"Rapor: {SAVE_DIR_ROOT}\n")

    all_rows = []
    for animal in ANIMALS:
        print(f"\n{'='*70}\n[*] HAYVAN: {animal.upper()}\n{'='*70}")
        loaders, class_names = build_loaders_for_animal(animal)
        if not loaders or not class_names:
            print(f"   [ATLA] {animal} için veri bulunamadı.")
            continue
        print(f"   Sınıflar ({len(class_names)}): {class_names}")
        for model_name, build_fn in MODEL_REGISTRY.items():
            all_rows.extend(evaluate_one(animal, model_name, build_fn, loaders, class_names))

    if not all_rows:
        print("\n[!] Hiçbir sonuç üretilemedi.")
        return

    # ---- ÖZET TABLO ----
    df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(SAVE_DIR_ROOT, "ozet_tum_sonuclar.csv")
    df.to_csv(summary_csv, index=False)

    print(f"\n\n{'='*70}\n  ÖZET — TEST split, Macro F1'e göre sıralı\n{'='*70}")
    test_df = df[df['split'] == 'test'].sort_values(['animal', 'macro_f1'], ascending=[True, False])
    if not test_df.empty:
        for animal in test_df['animal'].unique():
            sub = test_df[test_df['animal'] == animal]
            best = sub.iloc[0]
            print(f"\n  [{animal.upper()}]  En iyi: {best['model']}  (F1={best['macro_f1']:.4f})")
            for _, r in sub.iterrows():
                print(f"     {r['model']:18s}  Acc={r['accuracy']:.4f}  BalAcc={r['balanced_accuracy']:.4f}  "
                      f"F1={r['macro_f1']:.4f}  AUC={r['macro_auc']:.4f}  MCC={r['mcc']:.4f}  Kappa={r['cohen_kappa']:.4f}")

    print(f"\n✅ Bitti. Tüm grafikler/raporlar: {SAVE_DIR_ROOT}")
    print(f"   Özet CSV: {summary_csv}")


if __name__ == "__main__":
    main()
