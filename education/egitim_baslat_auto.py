import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification
import os
from tqdm import tqdm
import timm  # CoAtNet için: pip install timm

# --- AYARLAR ---
DATA_DIR = "./dataset_split/dataset_split"
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 5e-5
EARLY_STOP_PATIENCE = 7  # 7 epoch boyunca iyileşme olmazsa durdur

def get_next_version_name(base_name):
    """Dosya adını kontrol eder ve v1, v2... ekleyerek benzersiz isim döner."""
    counter = 1
    while True:
        filename = f"{base_name}_best_v{counter}.pth"
        if not os.path.exists(filename):
            return filename
        counter += 1

def train_loop(model, train_loader, val_loader, device, epochs, model_name):
    save_path = get_next_version_name(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # İYİLEŞTİRME: Başarı artmadığında LR'yi %50 düşürür
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_acc = 0.0
    patience_counter = 0

    print(f"\n===== {model_name} EĞİTİMİ BAŞLIYOR (Hedef: {save_path}) =====")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # HuggingFace / Timm çıktı uyumluluğu
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)

        # Doğrulama (Validation)
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

        val_acc = (corrects.double() / total).item()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} -> Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")

        # Learning Rate Güncelleme
        scheduler.step(val_acc)

        # En İyi Modeli Kaydetme ve Early Stopping Kontrolü
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> [KAYDEDİLDİ] Yeni en iyi başarı: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"!!! EARLY STOPPING: {model_name} gelişme göstermediği için durduruldu.")
                break

    print(f"===== {model_name} TAMAMLANDI. En İyi Başarı: {best_acc:.4f} =====\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    # Veri Dönüşümleri (Sizin tarafınızda sabitlenen boyutlar)
    NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

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

    # Veri Yükleyiciler
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transforms)

    num_classes = len(train_dataset.classes)
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # 1. ConvNeXt
    print("\n>>> ConvNeXt Başlatılıyor...")
    model_conv = ConvNextForImageClassification.from_pretrained(
        "facebook/convnext-base-224", num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)
    train_loop(model_conv, train_loader, val_loader, device, EPOCHS, "ConvNeXt")
    del model_conv
    torch.cuda.empty_cache()

    # 2. EfficientNet-B0
    print("\n>>> EfficientNet-B0 Başlatılıyor...")
    model_eff = EfficientNetForImageClassification.from_pretrained(
        "google/efficientnet-b0", num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)
    train_loop(model_eff, train_loader, val_loader, device, EPOCHS, "EfficientNet_B0")
    del model_eff
    torch.cuda.empty_cache()

    # 3. CoAtNet (Timm)
    print("\n>>> CoAtNet Başlatılıyor...")
    try:
        model_coat = timm.create_model('coatnet_0_rw_224', pretrained=True, num_classes=num_classes).to(device)
        train_loop(model_coat, train_loader, val_loader, device, EPOCHS, "CoAtNet")
        del model_coat
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"CoAtNet hatası: {e}. 'pip install timm' yüklü mü?")

if __name__ == '__main__':
    main()