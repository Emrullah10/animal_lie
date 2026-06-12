import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, EfficientNetForImageClassification, CvtForImageClassification
import os
from tqdm import tqdm
import timm
import random
import numpy as np
import argparse

# =========================================================
# 1. ARGPARSE AYARLARI (Modelleri Terminalden Seçme Eklendi)
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Hayvan Hastalıkları Sınıflandırma Modelleri Eğitimi")

    parser.add_argument("--data_dir", type=str, default="./dataset_split/dataset_split", help="Veri setinin ana klasörü")
    parser.add_argument("--batch_size", type=int, default=16, help="Aynı anda işlenecek fotoğraf sayısı")
    parser.add_argument("--epochs", type=int, default=30, help="Toplam eğitim döngüsü (Epoch)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Öğrenme oranı (Learning Rate)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping bekleme süresi")
    parser.add_argument("--seed", type=int, default=42, help="Tekrarlanabilirlik için sabit seed")

    # YENİ: Çalıştırılacak modelleri terminalden liste olarak alma
    parser.add_argument("--models", nargs="+", default=["cvt", "resnest"],
                        help="Çalıştırılacak modelleri yazın (Örn: --models cvt resnest convnext)")

    return parser.parse_args()

# =========================================================
# 2. SEED VE KAYIT FONKSİYONLARI
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_next_version_name(base_name):
    counter = 1
    while True:
        filename = f"{base_name}_best_v{counter}.pth"
        if not os.path.exists(filename):
            return filename
        counter += 1

# =========================================================
# 3. TRAIN LOOP (Eğitim Döngüsü)
# =========================================================
def train_loop(model, train_loader, valid_loader, device, model_name, class_weights, args):
    save_path = get_next_version_name(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("-> Loss fonksiyonu sınıf ağırlıkları ile (Dengeli) oluşturuldu.")
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_acc = 0.0
    patience_counter = 0

    print(f"\n===== {model_name} EĞİTİMİ BAŞLIYOR (Hedef: {save_path}) =====")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
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
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits

                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        valid_acc = (corrects.double() / total).item()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} -> Train Loss: {epoch_loss:.4f} | valid Acc: {valid_acc:.4f} | LR: {current_lr}")

        scheduler.step(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> [KAYDEDİLDİ] Yeni en iyi başarı: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"!!! EARLY STOPPING: {model_name} gelişme göstermediği için durduruldu.")
                break

    print(f"===== {model_name} TAMAMLANDI. En İyi Başarı: {best_acc:.4f} =====\n")

# =========================================================
# 4. MAIN FONKSİYONU
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transforms)
    valid_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=valid_test_transforms)

    num_classes = len(train_dataset.classes)

    class_counts = [0] * num_classes
    for label in train_dataset.targets:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Seçilen modelleri terminalden gelen argümana göre küçük harfe çevirip kontrol ediyoruz
    selected_models = [m.lower() for m in args.models]
    print(f"\nSeçilen Modeller: {selected_models}")

    # 1. CvT (Microsoft Base)
    if "cvt" in selected_models:
        print("\n>>> CvT Başlatılıyor...")
        model_cvt = CvtForImageClassification.from_pretrained(
            "microsoft/cvt-21", num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(device)
        train_loop(model_cvt, train_loader, valid_loader, device, "CvT", class_weights_tensor, args)
        del model_cvt
        torch.cuda.empty_cache()

    # 2. ResNeST (Timm üzerinden resmi Base)
    if "resnest" in selected_models:
        print("\n>>> ResNeST Başlatılıyor...")
        try:
            model_resnest = timm.create_model('resnest50d', pretrained=True, num_classes=num_classes).to(device)
            train_loop(model_resnest, train_loader, valid_loader, device, "ResNeST", class_weights_tensor, args)
            del model_resnest
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"ResNeST hatası: {e}")

    # İleride kullanmak istersen diye ConvNeXt ve EfficientNet'i de if bloğu içine aldım
    if "convnext" in selected_models:
        print("\n>>> ConvNeXt Başlatılıyor...")
        model_conv = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-base-224", num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(device)
        train_loop(model_conv, train_loader, valid_loader, device, "ConvNeXt", class_weights_tensor, args)
        del model_conv
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()