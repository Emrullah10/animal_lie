import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import json
import csv
import timm

# --- AYARLAR ---
DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR_ROOT = "test_sonuclari_detayli"  # Ana klasör
os.makedirs(SAVE_DIR_ROOT, exist_ok=True)

print(f"Kullanılan Cihaz: {DEVICE}")

# --- VERİ SETİ HAZIRLIĞI ---
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

try:
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    test_image_paths = [s[0] for s in test_dataset.samples]

    print(f"Test Verisi Yüklendi. Sınıflar ({num_classes}): {class_names}")
except Exception as e:
    print(f"HATA: Test veri seti yüklenemedi. ({e})")
    exit()

# --- YARDIMCI FONKSİYONLAR ---

def save_classification_report(report_dict, out_path):
    """JSON ve CSV olarak kaydeder"""
    with open(out_path + ".json", "w") as f:
        json.dump(report_dict, f, indent=2)

    rows = []
    for key, val in report_dict.items():
        if isinstance(val, dict):
            rows.append([key, val.get("precision",0), val.get("recall",0), val.get("f1-score",0), val.get("support",0)])

    with open(out_path + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for r in rows:
            writer.writerow(r)

def save_predictions_csv(image_paths, true_labels, pred_labels, pred_probs, class_names, out_path):
    """Detaylı tahmin listesi kaydeder"""
    true_names = [class_names[i] for i in true_labels]
    pred_names = [class_names[i] for i in pred_labels]
    max_probs = [np.max(p) for p in pred_probs]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dosya Yolu", "Gercek_Sinif", "Tahmin_Edilen", "Guven_Orani"])
        for p, t, pr, prob in zip(image_paths, true_names, pred_names, max_probs):
            writer.writerow([p, t, pr, f"{prob:.4f}"])

def plot_f1_bar(report_dict, class_names, out_dir, model_name):
    """F1 skor grafiği"""
    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names if cls in report_dict]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=f1_scores, palette='viridis')
    plt.title(f'{model_name} - Sınıf Bazlı F1 Skoru')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_f1_per_class.png"))
    plt.close()

# --- MODEL YÜKLEME ---

def load_model_safely(path, model_type, num_classes):
    print(f"\n>>> Model Yükleniyor: {path}")
    try:
        if model_type == "convnext":
            model = ConvNextForImageClassification.from_pretrained(
                "facebook/convnext-base-224", num_labels=num_classes, ignore_mismatched_sizes=True
            )
        elif model_type == "efficientnet":
            model = EfficientNetForImageClassification.from_pretrained(
                "google/efficientnet-b0", num_labels=num_classes, ignore_mismatched_sizes=True
            )
        elif model_type == "coatnet":
            model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=num_classes)

        model.to(DEVICE)

        state_dict = torch.load(path, map_location=DEVICE)

        # Sınıf sayısı kontrolü
        for key, val in state_dict.items():
            if ("classifier" in key or "head.fc" in key) and "weight" in key:
                if val.shape[0] != num_classes:
                    print(f"!!! HATA: Model {val.shape[0]} sınıf için eğitilmiş, veri setinde {num_classes} sınıf var.")
                    return None

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return None

# --- ANA ANALİZ FONKSİYONU ---

def evaluate_full_report(model, model_name):
    if model is None:
        return

    # --- YENİ EKLENEN KISIM: MODEL İÇİN ÖZEL KLASÖR OLUŞTURMA ---
    model_save_dir = os.path.join(SAVE_DIR_ROOT, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"\n{'='*20} {model_name} ANALİZİ BAŞLIYOR {'='*20}")
    print(f"📂 Sonuçlar kaydedilecek: {model_save_dir}")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Raporlar (Bu sefer model klasörünün içine kaydediyoruz)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    save_classification_report(report_dict, os.path.join(model_save_dir, f"{model_name}_report"))

    save_predictions_csv(test_image_paths, all_labels, all_preds, all_probs, class_names,
                         os.path.join(model_save_dir, f"{model_name}_predictions.csv"))

    # 2. Görselleştirmeler
    plot_f1_bar(report_dict, class_names, model_save_dir, model_name)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # ROC Curves
    y_test_bin = label_binarize(all_labels, classes=range(num_classes))
    fpr, tpr, roc_auc = dict(), dict(), dict()

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_save_dir, f"{model_name}_roc.png"))
    plt.close()

    # AUC Score JSON
    auc_dict = {class_names[i]: float(roc_auc[i]) for i in range(num_classes)}
    with open(os.path.join(model_save_dir, f"{model_name}_auc_scores.json"), "w") as f:
        json.dump(auc_dict, f, indent=2)

if __name__ == "__main__":
    # Model Dosya İsimleri (Eğer 'v2' ise burayı güncellemeyi unutma)
    path_conv = "ConvNeXt_best_v1.pth"
    path_eff = "EfficientNet_B0_best_v1.pth"
    path_coat = "CoAtNet_best_v1.pth"

    # 1. ConvNeXt
    if os.path.exists(path_conv):
        model_conv = load_model_safely(path_conv, "convnext", num_classes)
        evaluate_full_report(model_conv, "ConvNeXt")
    else:
        print(f"ATLANDI: {path_conv} bulunamadı.")

    # 2. EfficientNet
    if os.path.exists(path_eff):
        model_eff = load_model_safely(path_eff, "efficientnet", num_classes)
        evaluate_full_report(model_eff, "EfficientNet")
    else:
        print(f"ATLANDI: {path_eff} bulunamadı.")

    # 3. CoAtNet
    if os.path.exists(path_coat):
        model_coat = load_model_safely(path_coat, "coatnet", num_classes)
        evaluate_full_report(model_coat, "CoAtNet")
    else:
        print(f"ATLANDI: {path_coat} bulunamadı.")

    print(f"\n✅ TÜM İŞLEMLER BİTTİ. Sonuçlar '{SAVE_DIR_ROOT}' altında düzenli klasörlerde.")