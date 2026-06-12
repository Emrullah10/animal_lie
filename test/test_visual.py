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



DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16


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
    print(f"Sınıflar ({num_classes} adet): {class_names}")
except Exception as e:
    print(f"Hata: {e}")
    exit()



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

def visualize_performance(model, model_name):
    print(f"\n--- {model_name} Analiz Ediliyor ---")
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


    print(f"\n>>> {model_name} SINIFLANDIRMA RAPORU:")
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))




    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.show()


    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=f1_scores, palette='viridis')
    plt.title(f'{model_name} - Her Hastalık İçin F1 Başarısı')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Alarm)')
    plt.ylabel('True Positive Rate (Doğru Tespit)')
    plt.title(f'{model_name} - ROC Eğrileri (Her Sınıf İçin)')
    plt.legend(loc="lower right")
    plt.show()



path_conv = "ConvNeXt_best_v1.pth"
path_eff = "EfficientNet_B0_best_v1.pth"

if os.path.exists(path_conv):
    model = load_model(path_conv, "convnext")
    visualize_performance(model, "ConvNeXt")

if os.path.exists(path_eff):
    model = load_model(path_eff, "efficientnet")
    visualize_performance(model, "EfficientNet-B0")