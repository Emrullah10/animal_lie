import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



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
    print(f"Test seti yüklendi. Sınıflar: {class_names}")
except Exception as e:
    print(f"Hata: Test klasörü bulunamadı. {e}")
    exit()



def load_convnext(path):
    print(f"Yükleniyor: {path}")
    model = ConvNextForImageClassification.from_pretrained(
        "facebook/convnext-base-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_efficientnet(path):
    print(f"Yükleniyor: {path}")
    model = EfficientNetForImageClassification.from_pretrained(
        "google/efficientnet-b0",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, model_name):
    print(f"\n--- {model_name} Test Ediliyor ---")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())


    print(f"\n>>> {model_name} Sınıflandırma Raporu:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()




convnext_path = "ConvNeXt_best_v1.pth"
effnet_path = "EfficientNet_B0_best_v1.pth"

if os.path.exists(convnext_path):
    model_conv = load_convnext(convnext_path)
    evaluate_model(model_conv, "ConvNeXt")
else:
    print(f"UYARI: {convnext_path} bulunamadı.")

if os.path.exists(effnet_path):
    model_eff = load_efficientnet(effnet_path)
    evaluate_model(model_eff, "EfficientNet-B0")