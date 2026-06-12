import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification, CvtForImageClassification
import os
import numpy as np
import pandas as pd
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
SAVE_DIR_ROOT = "final_analiz_raporu"
os.makedirs(SAVE_DIR_ROOT, exist_ok=True)

# Gerekli dönüşümler
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Veri setini yükle
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)
test_image_paths = [s[0] for s in test_dataset.samples]

# --- GÖRSELLEŞTİRME FONKSİYONLARI ---

def plot_training_history(log_path, out_dir, model_name):
    """Eğitim sırasındaki Loss ve Accuracy grafiklerini çizer"""
    if not os.path.exists(log_path):
        print(f"⚠️ UYARI: {log_path} bulunamadı, eğitim grafikleri çizilemiyor.")
        return

    try:
        df = pd.read_csv(log_path)
        plt.figure(figsize=(14, 5))

        # Loss Grafiği
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], 'b-o', label='Eğitim Kaybı', markersize=4)
        plt.plot(df['epoch'], df['val_loss'], 'r-s', label='Doğrulama Kaybı', markersize=4)
        plt.title(f'{model_name} - Kayıp (Loss) Grafiği')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Accuracy Grafiği
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], 'g-o', label='Eğitim Başarısı', markersize=4)
        plt.plot(df['epoch'], df['val_acc'], 'm-s', label='Doğrulama Başarısı', markersize=4)
        plt.title(f'{model_name} - Doğruluk (Accuracy) Grafiği')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_training_history.png"))
        plt.close()
    except Exception as e:
            print(f"Grafik çizilirken hata: {e}")

def plot_detailed_metrics(all_labels, all_preds, all_probs, class_names, out_dir, model_name):
    """Test sonuçlarını görselleştirir (Confusion Matrix, ROC, F1)"""

    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # 2. ROC Curves
    y_test_bin = label_binarize(all_labels, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f'{model_name} - ROC Eğrileri')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, f"{model_name}_roc_curves.png"))
    plt.close()

# --- MODEL YÜKLEME VE ANALİZ ---

def evaluate_model(path, model_type, log_path, model_name):
    model_save_dir = os.path.join(SAVE_DIR_ROOT, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"\n🚀 {model_name} Analizi Başlıyor...")

    # Modeli Yükle
    try:
        if model_type == "convnext":
            model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224", num_labels=num_classes, ignore_mismatched_sizes=True)
        elif model_type == "efficientnet":
            model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0", num_labels=num_classes, ignore_mismatched_sizes=True)
        elif model_type == "coatnet":
            model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=num_classes)
        elif model_type == "resnest":
            model = timm.create_model('resnest50d', pretrained=False, num_classes=num_classes)
        elif model_type == "cvt":
            model = CvtForImageClassification.from_pretrained("microsoft/cvt-21", num_labels=num_classes, ignore_mismatched_sizes=True)

        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"❌ {model_name} yüklenemedi: {e}")
        return

    # Tahminleri Al
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Görselleştirmeleri Çalıştır ---
    # 1. Eğitim Geçmişi (Train Loss/Acc)
    plot_training_history(log_path, model_save_dir, model_name)

    # 2. Test Metrikleri
    plot_detailed_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs), class_names, model_save_dir, model_name)

    # 3. Raporu Kaydet
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(os.path.join(model_save_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print(f"✅ {model_name} tamamlandı. Sonuçlar: {model_save_dir}")

if __name__ == "__main__":
    # Dosya yolları ve log dosyaları (Eğitim sırasında kaydedilen CSV'ler)
    models_to_test = [
        {"path": "ConvNeXt_best_v1.pth", "type": "convnext", "log": "convnext_training_log.csv", "name": "ConvNeXt"},
        {"path": "EfficientNet_B0_best_v1.pth", "type": "efficientnet", "log": "effnet_training_log.csv", "name": "EfficientNet"},
        {"path": "CoAtNet_best_v1.pth", "type": "coatnet", "log": "coatnet_training_log.csv", "name": "CoAtNet"},
        {"path": "ResNeSt_best_v1.pth", "type": "resnest", "log": "resnest_training_log.csv", "name": "ResNeSt"},
        {"path": "CvT_best_v1.pth", "type": "cvt", "log": "cvt_training_log.csv", "name": "CvT"}
    ]

    for m in models_to_test:
        if os.path.exists(m["path"]):
            evaluate_model(m["path"], m["type"], m["log"], m["name"])
        else:
            print(f"Atlanıyor: {m['path']} bulunamadı.")