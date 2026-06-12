import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification
import os
from tqdm import tqdm
import sys







DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*40}")
    print(f"Kullanılan Cihaz: {device}")
    if device.type == 'cuda':
        print(f"Ekran Kartı Modeli: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Durumu: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB kullanılıyor.")
    else:
        print("UYARI: NVIDIA GPU bulunamadı! Eğitim CPU üzerinde çok yavaş olacaktır.")
    print(f"{'='*40}\n")



    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    print("Veri seti yükleniyor...")


    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"HATA: '{DATA_DIR}/train' klasörü bulunamadı!")
        print("Lütfen 'dataset_split' klasörünün kod dosyasıyla aynı yerde olduğundan emin olun.")
        return

    try:
        train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transforms)
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return


    num_classes = len(train_dataset.classes)
    print(f"Tespit Edilen Sınıf Sayısı: {num_classes}")
    print(f"Sınıflar: {train_dataset.classes}")


    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)




    print("\n>>> ConvNeXt Modeli Hazırlanıyor...")
    try:
        model_conv = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-base-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        model_conv.classifier = nn.Linear(model_conv.classifier.in_features, num_classes)
        model_conv.to(device)


        train_loop(model_conv, train_loader, val_loader, device, EPOCHS, "ConvNeXt")


        del model_conv
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"ConvNeXt hatası: {e}")




    print("\n>>> EfficientNet-B0 Modeli Hazırlanıyor...")

    try:
        model_eff = EfficientNetForImageClassification.from_pretrained(
            "google/efficientnet-b0",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        model_eff.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model_eff.classifier.in_features, num_classes)
        )
        model_eff.to(device)


        train_loop(model_eff, train_loader, val_loader, device, EPOCHS, "EfficientNet_B0")

    except Exception as e:
        print(f"EfficientNet hatası: {e}")

def train_loop(model, train_loader, val_loader, device, epochs, model_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    print(f"\n===== {model_name} EĞİTİMİ BAŞLIYOR =====")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0


        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)


        model.eval()
        corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_acc = corrects.double() / total
        print(f"Epoch {epoch+1} Tamamlandı -> Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")


        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"{model_name}_best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> En iyi model kaydedildi: {save_path}")

    print(f"===== {model_name} EĞİTİMİ BİTTİ. En Yüksek Başarı: {best_acc:.4f} =====")

if __name__ == '__main__':
    main()