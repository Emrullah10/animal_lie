#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIM 3 — Augmentation (dinamik hedef)
Çalıştır: python adim3_augment.py

Hedef mantığı:
  n < 300       →  300'e tamamla
  300 ≤ n < 400 →  400'e tamamla
  400 ≤ n       →  dokunma (zaten yeterli)

Kurulum (bir kere):
  pip install pillow numpy tqdm
"""

import random
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aug")

# ═════════════════════════════════════════════════
# AYARLAR
# ═════════════════════════════════════════════════

KAYNAK = r"C:\Users\kemal\PycharmProjects\cnn\dengeli_ham"
HEDEF  = r"C:\Users\kemal\PycharmProjects\cnn\dengeli_augmented"
SEED   = 42
IMG_SIZE = (224, 224)

EYE_CLASSES = {
    "dog_blepharitis", "dog_entropion",
    "cat_gingivitis", "cat_stomatitis", "cat_tooth_resorption",
}

# ─────────────────────────────────────────────────
# Dinamik hedef fonksiyonu — sadece buraya bakman yeter
# ─────────────────────────────────────────────────
def hedef_bul(n: int) -> int:
    if n < 300:
        return 300      # 300'e tamamla
    elif n < 400:
        return 400      # 400'e tamamla
    else:
        return n        # dokunma

# ═════════════════════════════════════════════════
# AUGMENTATION FONKSİYONLARI
# ═════════════════════════════════════════════════

def aug_hflip(img, _ref=None):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def aug_rotate(img, _ref=None):
    return img.rotate(random.uniform(-12, 12),
                      resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

def aug_brightness(img, _ref=None):
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))

def aug_contrast(img, _ref=None):
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.20))

def aug_noise(img, _ref=None):
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 8, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def aug_cutmix(img, ref_img):
    if ref_img is None:
        return img
    w, h = img.size
    ref  = ref_img.resize((w, h), Image.BILINEAR)
    lam  = np.clip(np.random.beta(1.0, 1.0), 0.3, 0.7)
    cw   = int(w * np.sqrt(1 - lam))
    ch   = int(h * np.sqrt(1 - lam))
    cx   = random.randint(0, w - cw)
    cy   = random.randint(0, h - ch)
    result = img.copy()
    result.paste(ref.crop((cx, cy, cx + cw, cy + ch)), (cx, cy))
    return result

FULL_PIPELINE = [
    (aug_hflip,      "hflip",    False),
    (aug_rotate,     "rot",      False),
    (aug_brightness, "bright",   False),
    (aug_contrast,   "contrast", False),
    (aug_noise,      "noise",    False),
    (aug_cutmix,     "cutmix",   True),
]
EYE_PIPELINE = [x for x in FULL_PIPELINE if not x[2]]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ═════════════════════════════════════════════════
# ANA DÖNGÜ
# ═════════════════════════════════════════════════

random.seed(SEED)
np.random.seed(SEED)

src_root = Path(KAYNAK)
dst_root = Path(HEDEF)
report   = {"timestamp": datetime.now().isoformat(), "classes": {}}

log.info("=" * 58)
log.info("  ADIM 3 — DİNAMİK HEDEFLE AUGMENTATION")
log.info("  < 300  →  300 | 300–399  →  400 | ≥ 400  →  değişmez")
log.info("=" * 58)

for sinif_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
    sinif = sinif_dir.name
    paths = [p for p in sinif_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not paths:
        continue

    out_dir = dst_root / sinif
    out_dir.mkdir(parents=True, exist_ok=True)

    # Görselleri yükle
    images = {}
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
            images[p] = img
        except Exception:
            continue

    if not images:
        continue

    n_orig = len(images)
    hedef  = hedef_bul(n_orig)
    need   = hedef - n_orig

    # Orijinalleri kaydet
    for p, img in images.items():
        img.save(out_dir / p.name, quality=92)

    if need <= 0:
        log.info(f"  {sinif:<32} {n_orig:>4}  →  dokunulmadı")
        report["classes"][sinif] = {
            "original": n_orig, "augmented": 0,
            "total": n_orig, "hedef": hedef
        }
        continue

    # Augmentation
    pipeline   = EYE_PIPELINE if sinif in EYE_CLASSES else FULL_PIPELINE
    orig_paths = list(images.keys())
    aug_count  = 0
    round_num  = 0

    log.info(f"  {sinif:<32} {n_orig:>4}  →  hedef {hedef}  (+{need} aug)")

    with tqdm(total=need, desc=f"  {sinif}", leave=False) as pbar:
        while aug_count < need:
            fn, name, needs_ref = pipeline[round_num % len(pipeline)]
            round_num += 1
            shuffled = orig_paths.copy()
            random.shuffle(shuffled)

            for src_path in shuffled:
                if aug_count >= need:
                    break
                ref_img = None
                if needs_ref:
                    kand = [p for p in orig_paths if p != src_path]
                    if kand:
                        ref_img = images[random.choice(kand)]
                try:
                    aug_img = fn(images[src_path], ref_img)
                except Exception:
                    continue
                out_name = f"{src_path.stem}_{name}_{aug_count:04d}{src_path.suffix}"
                aug_img.save(out_dir / out_name, quality=92)
                aug_count += 1
                pbar.update(1)

    report["classes"][sinif] = {
        "original": n_orig, "augmented": aug_count,
        "total": n_orig + aug_count, "hedef": hedef
    }

# Rapor
dst_root.mkdir(parents=True, exist_ok=True)
with open(dst_root / "aug_raporu.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

log.info("=" * 58)
log.info("  TAMAMLANDI")
for s, d in report["classes"].items():
    log.info(f"  {s:<32} {d['original']:>4} → {d['total']:>4}")
log.info("=" * 58)
print("\nŞimdi adim4_split.py çalıştırabilirsin.")