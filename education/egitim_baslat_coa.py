import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification
import os
from tqdm import tqdm
import sys
import timm  # CoAtNet için gerekli kütüphane (pip install timm)

DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 5e-5

def get_next_version_name(base_name):
    """
    Dosya adını kontrol eder ve üzerine yazmamak için v1, v2, v3... ekler.
    Örn: CoAtNet_best_v1.pth, CoAtNet_best_v2.pth
    """
    counter = 1
    while True:
        filename = f"{base_name}_best_v{counter}.pth"
        if not os.path.exists(filename):
            return filename
        counter += 1

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

    # ---------------------------------------------------------
    # 1. ConvNeXt Modeli
    # ---------------------------------------------------------
    print("\n>>> ConvNeXt Modeli Hazırlanıyor...")
    try:
        model_conv = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-base-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # HuggingFace modellerinde classifier katmanını güncelleme
        model_conv.classifier = nn.Linear(model_conv.classifier.in_features, num_classes)
        model_conv.to(device)

        train_loop(model_conv, train_loader, val_loader, device, EPOCHS, "ConvNeXt")

        del model_conv
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"ConvNeXt hatası: {e}")

    # ---------------------------------------------------------
    # 2. EfficientNet-B0 Modeli
    # ---------------------------------------------------------
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

        del model_eff
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"EfficientNet hatası: {e}")

    # ---------------------------------------------------------
    # 3. CoAtNet Modeli (YENİ EKLENDİ)
    # ---------------------------------------------------------
    print("\n>>> CoAtNet Modeli Hazırlanıyor (timm kütüphanesi ile)...")
    try:
        # coatnet_0_rw_224 yaygın kullanılan, dengeli bir versiyondur.
        # pretrained=True ile ImageNet ağırlıklarını çeker.
        # num_classes parametresi son katmanı otomatik ayarlar.
        model_coat = timm.create_model('coatnet_0_rw_224', pretrained=True, num_classes=num_classes)

        model_coat.to(device)

        train_loop(model_coat, train_loader, val_loader, device, EPOCHS, "CoAtNet")

        del model_coat
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"CoAtNet hatası: {e}")
        print("Lütfen 'pip install timm' komutu ile kütüphaneyi kurduğunuzdan emin olun.")


def train_loop(model, train_loader, val_loader, device, epochs, model_name):
    # Kayıt edilecek dosya adını versiyon kontrolü ile belirle
    save_path = get_next_version_name(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    print(f"\n===== {model_name} EĞİTİMİ BAŞLIYOR (Hedef Dosya: {save_path}) =====")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # HuggingFace modelleri .logits döndürürken, timm modelleri direkt Tensor döndürür.
            # Bu ayrımı yapmak için kontrol ekliyoruz:
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

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

                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits

                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_acc = corrects.double() / total
        print(f"Epoch {epoch+1} Tamamlandı -> Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            # Belirlenen versiyonlu isme kaydet
            torch.save(model.state_dict(), save_path)
            print(f"--> En iyi model kaydedildi: {save_path}")

    print(f"===== {model_name} EĞİTİMİ BİTTİ. En Yüksek Başarı: {best_acc:.4f} =====")

if __name__ == '__main__':
    main()