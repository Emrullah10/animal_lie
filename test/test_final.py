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

# --- AYARLAR ---
DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "test_sonuclari_final"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    print(f"Test Verisi Yüklendi. Sınıflar ({num_classes}): {class_names}")
except Exception as e:
    print(f"HATA: Test veri seti yüklenemedi. Klasör yolunu kontrol et. ({e})")
    exit()

def load_model_safely(path, model_type, num_classes):
    print(f"\n>>> Model Yükleniyor: {path}")

    # 1. Modeli Mimarisiyle Başlat
    try:
        if model_type == "convnext":
            model = ConvNextForImageClassification.from_pretrained(
                "facebook/convnext-base-224",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            # Eğitim kodunda ekstra bir oynama yapmadıysan burayı elleme!
            # HuggingFace zaten sınıf sayısına göre classifier oluşturur.

        elif model_type == "efficientnet":
            model = EfficientNetForImageClassification.from_pretrained(
                "google/efficientnet-b0",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            # DİKKAT: Test kodundaki manuel 'nn.Sequential' eklemesi kaldırıldı.
            # Eğitim kodunda bu yoktu, bu yüzden hata alıyordun.

        model.to(DEVICE)

        # 2. State Dict Yüklemesi (Boyut Kontrolü ile)
        state_dict = torch.load(path, map_location=DEVICE)

        # State dict içindeki classifier boyutlarını kontrol et
        # Bu kısım ConvNeXt hatasını anlamanı sağlar
        for key, val in state_dict.items():
            if "classifier" in key and "weight" in key:
                if val.shape[0] != num_classes:
                    print(f"!!! KRİTİK UYARI !!!")
                    print(f"Yüklenen model {val.shape[0]} sınıf için eğitilmiş.")
                    print(f"Ancak şu anki veri setin {num_classes} sınıf içeriyor.")
                    print("Lütfen dataseti değiştirdikten sonra modeli YENİDEN EĞİTİN.")
                    return None

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None

def evaluate_and_plot(model, model_name):
    if model is None:
        print(f"{model_name} yüklenemediği için atlanıyor.")
        return

    print(f"\n{'='*20} {model_name} ANALİZ EDİLİYOR {'='*20}")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).logits
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Classification Report
    print(f"\n>>> {model_name} Sınıflandırma Raporu:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_CM.png"))
    plt.show()

    # 3. ROC Curves
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
    plt.title(f'{model_name} - ROC Eğrileri')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_ROC.png"))
    plt.show()

if __name__ == "__main__":
    # Dosya yollarını kontrol et. Eğer yeniden eğittiysen dosya adı v2, v3 olabilir!
    path_conv = "ConvNeXt_best_v1.pth"
    path_eff = "EfficientNet_B0_best_v1.pth"

    # Modelleri Yükle ve Test Et
    if os.path.exists(path_conv):
        model_conv = load_model_safely(path_conv, "convnext", num_classes)
        evaluate_and_plot(model_conv, "ConvNeXt")
    else:
        print(f"UYARI: {path_conv} bulunamadı.")

    if os.path.exists(path_eff):
        model_eff = load_model_safely(path_eff, "efficientnet", num_classes)
        evaluate_and_plot(model_eff, "EfficientNet")

    print(f"\n✅ Tüm işlemler tamamlandı. Grafikler '{SAVE_DIR}' klasöründe.")