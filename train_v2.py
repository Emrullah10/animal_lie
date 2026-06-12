import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    resnext101_32x8d, ResNeXt101_32X8D_Weights
)
from torch.utils.data import DataLoader
from transformers import ConvNextV2ForImageClassification, CvtForImageClassification
import timm
import os
import random
import numpy as np
import argparse
import multiprocessing
from tqdm import tqdm

# =========================================================
# MONKEY PATCH FOR CONVNEXTV2 GRN (Windows/CUDA FP16 Bug Fix)
# =========================================================
try:
    import transformers.models.convnextv2.modeling_convnextv2 as modeling_convnextv2
    
    def patched_grn_forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        input_dtype = hidden_states.dtype
        global_features = torch.linalg.vector_norm(
            hidden_states.to(torch.float32), ord=2, dim=(1, 2), keepdim=True
        ).to(input_dtype)
        norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        return hidden_states

    modeling_convnextv2.ConvNextV2GRN.forward = patched_grn_forward
    print("   [PATCH] ConvNextV2 GRN monkey patch applied successfully.")
except Exception as e:
    print(f"   [PATCH WARNING] ConvNextV2 GRN patch failed: {e}")

RUN_DIR = None

# =========================================================
# 1. ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Hayvan Hastalıkları Sınıflandırma - Train")
    parser.add_argument("--data_dir",          type=str,   default="./dataset_split/dataset_split")
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--epochs",             type=int,   default=30)
    parser.add_argument("--lr",                 type=float, default=5e-5,  help="Head (son katman) öğrenme oranı")
    parser.add_argument("--backbone_lr_mult",   type=float, default=0.1,   help="Backbone LR = lr * bu değer")
    parser.add_argument("--freeze_epochs",      type=int,   default=3,     help="Kaç epoch backbone dondurulsun")
    parser.add_argument("--warmup_epochs",      type=int,   default=3,     help="Unfreeze sonrası kaç epoch warmup")
    parser.add_argument("--label_smoothing",    type=float, default=0.1,   help="Label smoothing (0=kapalı)")
    parser.add_argument("--patience",           type=int,   default=7,     help="Early stopping sabrı")
    parser.add_argument("--clip_grad",          type=float, default=1.0)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument(
        "--models", nargs="+",
        default=["convnextv2", "efficientnetv2", "cvt", "resnext", "resnest"],
        help="Seçenekler: convnextv2 efficientnetv2 cvt resnext resnest"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Eğitime kalınan checkpoint klasöründen veya en son klasörden devam et ('latest' veya klasör yolu)"
    )
    return parser.parse_args()


# =========================================================
# 2. YARDIMCI FONKSİYONLAR
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # RTX 3060 (Ampere/8.6): Tensor Core hızlandırması için TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Ampere+ ve bf16 destekliyse autocast'te bf16 kullan (fp16'dan stabil ve hızlı)
AMP_DTYPE = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)

def get_model_save_paths(base_name, resume=False):
    # Tüm sınıfların eğitildiği genel model ağırlıklarını checkpoints/<run_dir>/ altına kaydeder
    save_dir = os.path.join("checkpoints", RUN_DIR or "run_general")
    os.makedirs(save_dir, exist_ok=True)

    if resume:
        counter = 1
        found_counter = 1
        while True:
            best_path = os.path.join(save_dir, f"{base_name}_best_v{counter}.pth")
            latest_path = os.path.join(save_dir, f"{base_name}_latest_v{counter}.pth")
            if os.path.exists(best_path) or os.path.exists(latest_path):
                found_counter = counter
                counter += 1
            else:
                break
        best_path = os.path.join(save_dir, f"{base_name}_best_v{found_counter}.pth")
        latest_path = os.path.join(save_dir, f"{base_name}_latest_v{found_counter}.pth")
        return best_path, latest_path
    else:
        counter = 1
        while True:
            best_path = os.path.join(save_dir, f"{base_name}_best_v{counter}.pth")
            latest_path = os.path.join(save_dir, f"{base_name}_latest_v{counter}.pth")
            if not os.path.exists(best_path) and not os.path.exists(latest_path):
                return best_path, latest_path
            counter += 1

def is_head_param(param_name, model_key):
    """Parametre adına göre head (son katman) mı backbone mı olduğunu döner."""
    head_keywords = {
        "convnextv2":     ["classifier"],
        "efficientnetv2": ["classifier"],
        "cvt":            ["classifier"],
        "resnext":        ["fc"],
        "resnest":        ["fc"],
    }
    keywords = head_keywords.get(model_key, ["classifier", "fc", "head"])
    return any(kw in param_name for kw in keywords)


# =========================================================
# 3. MODEL KURULUM
# =========================================================
def build_convnextv2(num_classes):
    return ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-base-22k-224",
        num_labels=num_classes, ignore_mismatched_sizes=True)

def build_efficientnetv2(num_classes):
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def build_cvt(num_classes):
    return CvtForImageClassification.from_pretrained(
        "microsoft/cvt-21",
        num_labels=num_classes, ignore_mismatched_sizes=True)

def build_resnext(num_classes):
    model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_resnest(num_classes):
    return timm.create_model('resnest101e', pretrained=True, num_classes=num_classes)


# =========================================================
# 4. FREEZE / UNFREEZE
# =========================================================
def freeze_backbone(model, model_key):
    """Sadece head (son katman) eğitilir, backbone dondurulur."""
    frozen = 0
    for name, param in model.named_parameters():
        if is_head_param(name, model_key):
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += 1
    print(f"   [FREEZE] {frozen} backbone parametresi donduruldu, sadece head eğitiliyor.")

def unfreeze_all(model):
    """Tüm parametreler açılır."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"   [UNFREEZE] Tüm parametreler açıldı. Toplam: {total/1e6:.1f}M")


# =========================================================
# 5. OPTİMİZATÖR (Backbone vs Head farklı LR)
# =========================================================
def build_optimizer(model, model_key, head_lr, backbone_lr):
    head_params     = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_head_param(name, model_key):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'})
    if head_params:
        param_groups.append({'params': head_params,     'lr': head_lr,     'name': 'head'})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    print(f"   [LR] Head: {head_lr:.2e}  |  Backbone: {backbone_lr:.2e}")
    return optimizer


# =========================================================
# 6. WARMUP (Linear)
# =========================================================
def apply_warmup_lr(optimizer, current_epoch, warmup_epochs, head_lr, backbone_lr):
    """Warmup süresince LR'yi doğrusal olarak artırır."""
    scale = (current_epoch + 1) / warmup_epochs
    for group in optimizer.param_groups:
        base = head_lr if group.get('name') == 'head' else backbone_lr
        group['lr'] = base * scale


# =========================================================
# 7. EĞİTİM DÖNGÜSÜ
# =========================================================
def train_loop(model, model_key, train_loader, val_loader, device, model_name, class_weights, args):
    best_path, latest_path = get_model_save_paths(model_name, resume=args.resume is not None)
    save_path = best_path
    scaler     = GradScaler('cuda')
    head_lr    = args.lr
    backbone_lr= args.lr * args.backbone_lr_mult

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    print(f"   [OK] Loss: Class-weighted CrossEntropy | Label Smoothing: {args.label_smoothing}")

    best_acc         = 0.0
    patience_counter = 0
    start_epoch      = 0
    warmup_done      = False

    # Check for resume checkpoint
    checkpoint = None
    if args.resume and os.path.exists(latest_path):
        print(f"   [RESUME] Checkpoint yükleniyor: {latest_path}")
        try:
            checkpoint = torch.load(latest_path, map_location=device)
            saved_epoch = checkpoint['epoch']
            early_stopped = checkpoint.get('early_stopped', False)
            
            if saved_epoch >= args.epochs - 1 or early_stopped:
                print(f"   [RESUME INFO] {model_name} eğitimi zaten tamamlanmış (Epoch: {saved_epoch+1}/{args.epochs}, Erken Durma: {early_stopped}). Geçiliyor...")
                return
            
            model.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint['best_acc']
            patience_counter = checkpoint['patience_counter']
            start_epoch = saved_epoch + 1
            print(f"   [RESUME OK] Model yükleme başarılı. Epoch {start_epoch+1} konumundan devam edilecek. En iyi Val Acc: {best_acc:.4f}")
        except Exception as e:
            print(f"   [RESUME HATA] Checkpoint yüklenirken hata oluştu (Sıfırdan başlanıyor): {e}")

    print(f"\n{'='*60}")
    print(f"  {model_name} EGITIMI BASLIYOR  ->  {save_path}")
    print(f"  Freeze: {args.freeze_epochs} epoch  |  Warmup: {args.warmup_epochs} epoch")
    print(f"{'='*60}")

    # ---- PHASE 1 / 2 Seçimi ----
    if start_epoch < args.freeze_epochs:
        print(f"\n  [PHASE 1] Backbone donduruldu ({args.freeze_epochs} epoch)")
        freeze_backbone(model, model_key)
        optimizer = build_optimizer(model, model_key, head_lr, backbone_lr)
    else:
        print(f"\n  [PHASE 2] Tüm katmanlar açıldı (Resume sonrası)")
        unfreeze_all(model)
        optimizer = build_optimizer(model, model_key, head_lr / 10, backbone_lr / 10)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5)

    # Load optimizer, scheduler and scaler state if resuming
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("   [RESUME OK] Optimizer, Scheduler ve Scaler durumları yüklendi.")
        except Exception as e:
            print(f"   [RESUME UYARI] Optimizer/Scheduler durumları yüklenemedi (Sıfır durumla devam ediliyor): {e}")

    for epoch in range(start_epoch, args.epochs):

        # Phase geçişi: freeze_epochs dolunca unfreeze + yeni optimizer
        if epoch == args.freeze_epochs and epoch > start_epoch:
            print(f"\n  [PHASE 2] Tüm katmanlar açıldı — Warmup başlıyor ({args.warmup_epochs} epoch)")
            unfreeze_all(model)
            optimizer = build_optimizer(model, model_key, head_lr / 10, backbone_lr / 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3, factor=0.5)

        # Warmup: Phase 2'nin ilk warmup_epochs epoch'u
        in_warmup = (epoch >= args.freeze_epochs and
                     epoch < args.freeze_epochs + args.warmup_epochs)
        if in_warmup:
            warmup_ep = epoch - args.freeze_epochs
            apply_warmup_lr(optimizer, warmup_ep, args.warmup_epochs, head_lr, backbone_lr)

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        phase_label = "FREEZE" if epoch < args.freeze_epochs else \
                      ("WARMUP" if in_warmup else "TRAIN")

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:02d}/{args.epochs} [{phase_label}]",
                    unit="batch")
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=AMP_DTYPE):
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)

        # ---- VALIDATION ----
        model.eval()
        corrects = 0
        total    = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                with autocast('cuda', dtype=AMP_DTYPE):
                    outputs = model(inputs)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                _, preds  = torch.max(outputs, 1)
                corrects  += torch.sum(preds == labels).item()
                total     += labels.size(0)

        val_acc    = corrects / total
        head_current_lr = next(
            g['lr'] for g in optimizer.param_groups if g.get('name') == 'head')
        print(f"  Epoch {epoch+1:02d} -> Loss: {epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Head LR: {head_current_lr:.2e}")

        # Warmup süresince scheduler adımı atma
        if not in_warmup:
            scheduler.step(val_acc)

        # ---- EN İYİ MODEL & EARLY STOPPING ----
        early_stop_triggered = False
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> [KAYDEDILDI] Yeni en iyi: {best_acc:.4f}  ({save_path})")
            patience_counter = 0
        else:
            # Freeze aşamasında early stopping sayma
            if epoch >= args.freeze_epochs:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  !!! EARLY STOPPING: {args.patience} epoch boyunca gelişme yok.")
                    early_stop_triggered = True

        # Her epoch sonunda en son durumu kaydet
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
            'early_stopped': early_stop_triggered
        }
        torch.save(checkpoint_state, latest_path)

        if early_stop_triggered:
            break

    print(f"\n  {model_name} TAMAMLANDI -> En Iyi Val Acc: {best_acc:.4f}")
    print(f"  Kaydedilen: {save_path}")
    print(f"{'='*60}\n")


# =========================================================
# 8. MAIN
# =========================================================
def main():
    global RUN_DIR
    import datetime

    args = parse_args()
    set_seed(args.seed)

    if args.resume:
        if args.resume.lower() == 'latest':
            checkpoints_dir = "checkpoints"
            if os.path.exists(checkpoints_dir):
                subdirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith("run_general_")]
                if subdirs:
                    subdirs.sort()
                    latest_dir = subdirs[-1]
                    RUN_DIR = latest_dir
                    print(f"   [RESUME] En son eğitim klasörü bulundu ve seçildi: {RUN_DIR}")
                else:
                    print("   [RESUME UYARI] 'run_general_' ile başlayan hiçbir klasör bulunamadı. Yeni eğitim başlatılıyor.")
                    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    RUN_DIR = f"run_general_{run_time}"
            else:
                print("   [RESUME UYARI] 'checkpoints' klasörü bulunamadı. Yeni eğitim başlatılıyor.")
                run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                RUN_DIR = f"run_general_{run_time}"
        else:
            resume_path = args.resume
            if os.path.isdir(resume_path):
                RUN_DIR = os.path.basename(os.path.normpath(resume_path))
            elif os.path.isdir(os.path.join("checkpoints", resume_path)):
                RUN_DIR = resume_path
            else:
                print(f"   [RESUME HATA] Belirtilen klasör bulunamadı: {resume_path}. Yeni eğitim başlatılıyor.")
                run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                RUN_DIR = f"run_general_{run_time}"
            print(f"   [RESUME] Belirtilen klasörden devam ediliyor: {RUN_DIR}")
    else:
        run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        RUN_DIR = f"run_general_{run_time}"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU"
    print(f"\nCihaz: {device} ({gpu_name})")
    print(f"Seçilen Modeller: {args.models}\n")

    # ---- TRANSFORMS ----
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # ---- DATASET ----
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data_dir, 'val'),   transform=val_transforms)

    num_classes = len(train_dataset.classes)
    print(f"Sınıf Sayısı: {num_classes}")
    print(f"Sınıflar: {train_dataset.classes}\n")

    # ---- CLASS WEIGHTS ----
    class_counts  = torch.zeros(num_classes)
    for label in train_dataset.targets:
        class_counts[label] += 1
    class_weights = (class_counts.sum() / (num_classes * class_counts)).to(device)

    # ---- DATALOADER ----
    num_workers = min(8, multiprocessing.cpu_count()) if device.type == 'cuda' else 0
    pin_memory  = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=4 if num_workers > 0 else None)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=4 if num_workers > 0 else None)

    # ---- MODEL KAYIT DEFTERİ ----
    model_registry = {
        "convnextv2":     ("ConvNeXtV2-Base",   build_convnextv2),
        "efficientnetv2": ("EfficientNetV2-M",  build_efficientnetv2),
        "cvt":            ("CvT-21",            build_cvt),
        "resnext":        ("ResNeXt101-32x8d",  build_resnext),
        "resnest":        ("ResNeSt101e",       build_resnest),
    }

    selected = [m.lower() for m in args.models]

    for key in selected:
        if key not in model_registry:
            print(f"[UYARI] Tanınmayan model: '{key}' — atlanıyor.")
            continue

        display_name, build_fn = model_registry[key]
        print(f"\n>>> {display_name} başlatılıyor...")

        try:
            model = build_fn(num_classes).to(device, memory_format=torch.channels_last)
            train_loop(model, key, train_loader, val_loader,
                       device, display_name, class_weights, args)
        except Exception as e:
            print(f"[HATA] {display_name}: {e}")
            import traceback; traceback.print_exc()
        finally:
            try: del model
            except NameError: pass
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()