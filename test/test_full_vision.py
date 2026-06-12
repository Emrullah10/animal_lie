import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             matthews_corrcoef, cohen_kappa_score, accuracy_score, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
from itertools import cycle



DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16



SAVE_DIR = "sonuc_grafikleri"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")


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
    print(f"Sınıflar: {class_names}")
except Exception as e:
    print(f"Hata: {e}")
    exit()


def print_ascii_bar(label, score, color_code="\033[92m"):
    """Terminalde metrikleri çubuk grafik olarak basar"""
    bar_len = int(score * 20)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    reset = "\033[0m"
    print(f"{label.ljust(15)} : {color_code}{bar}{reset} {score:.4f}")


def load_model(path, model_type):
    print(f"\n>>> Model Yükleniyor: {path}")
    if model_type == "convnext":
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224", num_labels=num_classes, ignore_mismatched_sizes=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_type == "efficientnet":
        model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0", num_labels=num_classes, ignore_mismatched_sizes=True)
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(model.classifier.in_features, num_classes))

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def analyze_model(model, model_name):
    print(f"\n{'='*20} {model_name} ANALİZİ {'='*20}")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).logits
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)




    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n>>> [TERMINAL] Sınıf Bazlı F1-Skorları:")
    for cls in class_names:
        score = report[cls]['f1-score']
        print_ascii_bar(cls, score)




    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    print(f"\n>>> [TERMINAL] Genel Performans Metrikleri:")
    print_ascii_bar("Accuracy", acc, "\033[94m")
    print_ascii_bar("Balanced Acc", balanced_acc, "\033[96m")
    print_ascii_bar("MCC Score", mcc, "\033[93m")
    print_ascii_bar("Kappa Score", kappa, "\033[95m")

    print(f"\nNOT: MCC ve Kappa, dengesiz veri setleri için en güvenilir metriklerdir.")






    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_ConfusionMatrix.png"))
    print(f"-> Grafik Kaydedildi: {model_name}_ConfusionMatrix.png")
    plt.close()


    precision_vals = [report[cls]['precision'] for cls in class_names]
    recall_vals = [report[cls]['recall'] for cls in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, precision_vals, width, label='Precision', color='skyblue')
    plt.bar(x + width/2, recall_vals, width, label='Recall', color='orange')
    plt.ylabel('Skor')
    plt.title(f'{model_name} - Precision ve Recall Karşılaştırması')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_Precision_Recall.png"))
    print(f"-> Grafik Kaydedildi: {model_name}_Precision_Recall.png")
    plt.close()


    y_test_bin = label_binarize(all_labels, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (AUC = {1:0.2f})'.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Eğrileri')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_ROC.png"))
    print(f"-> Grafik Kaydedildi: {model_name}_ROC.png")
    plt.close()



if __name__ == "__main__":

    path_conv = "ConvNeXt_best_model.pth"
    path_eff = "EfficientNet_B0_best_model.pth"

    if os.path.exists(path_conv):
        model = load_model(path_conv, "convnext")
        analyze_model(model, "ConvNeXt")

    if os.path.exists(path_eff):
        model = load_model(path_eff, "efficientnet")
        analyze_model(model, "EfficientNet-B0")

    print(f"\n✅ Tüm analizler bitti! Grafikler '{SAVE_DIR}' klasöründe.")