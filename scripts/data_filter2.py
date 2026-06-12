#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════╗
║   GELİŞMİŞ DATASET TEMİZLEYİCİ  v2.2  — HER AÇIDAN ROTASYON YAMASI       ║
║                                                                          ║
║   Belirlenen STEP (derece) aralığıyla (örn: 15°) tüm çemberi tarar.      ║
║   Hem CNN özellik bankası hem de pHash için tüm bu ara açılar            ║
║   hesaplanır.                                                            ║
╚══════════════════════════════════════════════════════════════════════════╝

Kurulum:
    pip install torch torchvision pillow imagehash numpy tqdm
"""

import re
import shutil
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False
    print("[UYARI] imagehash bulunamadı → pip install imagehash")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cleaner_v22")

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation keyword / pattern listeleri
# ─────────────────────────────────────────────────────────────────────────────

AUG_KEYWORDS = {
    "aug","augment","augmented","augmentation",
    "rot","rotate","rotation","rotated","rotate90","rotate180","rotate270",
    "flip","flipped","flop","mirror","mirrored","hflip","vflip",
    "horizontal","vertical","noise","noisy","gaussian","blur","blurred",
    "crop","cropped","trans","translate","bright","brightness","contrast",
    "gamma","hue","saturation","color","jitter","colorjitter","zoom",
    "scale","shear","warp","distort","perspective","affine","elastic",
    "clahe","equalize","mixup","cutout","cutmix","mosaic",
    "copy","dup","duplicate","synthetic","generated","gen","fake",
}
AUG_PATTERNS = [
    r"_aug\d*", r"_rot_?\d+", r"_flip[_hvxy01]?", r"_[hv]flip",
    r"_[a-z]{1,4}\d{1,3}$", r"\(\d+\)$", r"_\d{1,4}$",
    r"copy\s*\d*$", r"_v\d+$", r"\d{6,}$",
]

CAMERA_HW_TAGS = {"Make","Model","LensMake","LensModel","BodySerialNumber"}
EDIT_SW = {
    "gimp","photoshop","pillow","pil","opencv","imagemagick","python",
    "pytorch","keras","tensorflow","imgaug","albumentations","augmentor",
    "skimage","kornia","torchvision",
}


# ═════════════════════════════════════════════════════════════════════════════
#  ★ ROTASYON-INVARIANT CNN ÖZELLİK BANKASI (HER AÇI) ★
# ═════════════════════════════════════════════════════════════════════════════

# HASSASİYET AYARI BURADA:
# 15 yaparsan 15, 30, 45... (26 varyasyon)
# 10 yaparsan 10, 20, 30... (38 varyasyon)
STEP = 15

ROTATION_TRANSFORMS = [
    lambda im: im,                                  # 0 derece (Orijinal)
    lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), # Yatay Yansıma
]

# STEP değerine göre tüm çemberi tarar
for angle in range(STEP, 360, STEP):
    ROTATION_TRANSFORMS.append(lambda im, a=angle: im.rotate(a, expand=False))

N_ROTS = len(ROTATION_TRANSFORMS)


# ═════════════════════════════════════════════════════════════════════════════
# Ön analiz aşamaları (dosya adı / EXIF / piksel)
# ═════════════════════════════════════════════════════════════════════════════

def filename_aug_score(path: Path) -> dict:
    name = path.stem.lower()
    reasons, points = [], 0
    parts = re.split(r"[_\-\s\.\+]", name)
    for p in parts:
        if p in AUG_KEYWORDS:
            points += 45
            reasons.append(f"aug keyword: '{p}'")
            break
    for pattern in AUG_PATTERNS:
        if re.search(pattern, name):
            points += 35
            reasons.append(f"aug pattern: `{pattern}`")
            break
    if len(name) > 55:
        points += 10
        reasons.append(f"uzun isim ({len(name)} kr)")
    if re.fullmatch(r"\d+", name):
        points += 10
        reasons.append("yalnızca rakam")
    return {"confidence": min(points / 100.0, 1.0), "reasons": reasons}


def exif_aug_score(path: Path, img: Image.Image) -> dict:
    from PIL.ExifTags import TAGS
    reasons, points = [], 0
    exif_data = {}
    try:
        raw = img._getexif() if hasattr(img, "_getexif") else None
        if raw:
            exif_data = {TAGS.get(k, str(k)): v for k, v in raw.items()}
    except Exception:
        pass
    if not exif_data:
        points += 30
        reasons.append("EXIF yok")
    else:
        if any(t in exif_data for t in CAMERA_HW_TAGS):
            points -= 25
            reasons.append("kamera donanım EXIF mevcut")
        else:
            points += 15
            reasons.append("kamera donanım EXIF yok")
        sw = str(exif_data.get("Software","")).lower()
        if sw and any(s in sw for s in EDIT_SW):
            points += 30
            reasons.append(f"edit yazılımı: '{sw}'")
        dt_o = exif_data.get("DateTimeOriginal")
        dt_m = exif_data.get("DateTime")
        if dt_o and dt_m and dt_o != dt_m:
            points += 10
            reasons.append("zaman damgası uyuşmazlığı")
        if "GPSInfo" in exif_data:
            points -= 10
            reasons.append("GPS mevcut → orijinal")
    sz = path.stat().st_size / 1024
    if sz < 4:    points += 25; reasons.append(f"çok küçük ({sz:.1f}KB)")
    elif sz < 15: points += 10; reasons.append(f"küçük ({sz:.1f}KB)")
    elif sz > 300: points -= 10; reasons.append(f"büyük ({sz:.0f}KB)")
    return {"confidence": max(0.0, min(points / 100.0, 1.0)), "reasons": reasons}


def pixel_artifact_score(img: Image.Image) -> dict:
    reasons, points = [], 0
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    bw = max(4, min(h, w) // 18)

    for side, reg in [("üst", arr[:bw]), ("alt", arr[-bw:]),
                      ("sol", arr[:, :bw]), ("sağ", arr[:, -bw:])]:
        std = reg.std()
        if std < 1.5:
            points += 40; reasons.append(f"uniform kenar ({side}, std={std:.2f})"); break
        elif std < 4.0:
            points += 18; reasons.append(f"düşük varyanslı kenar ({side})"); break

    for c in range(3):
        ch = arr[:, :, c]; tot = ch.size
        if (ch < 3).sum()/tot > 0.04 and (ch > 252).sum()/tot > 0.04:
            points += 22; reasons.append("histogram kırpma"); break

    if h >= 20 and w >= 20:
        qs = [arr[:h//2,:w//2], arr[:h//2,w//2:], arr[h//2:,:w//2], arr[h//2:,w//2:]]
        vs = [np.var(np.mean(q,axis=2)) for q in qs]
        mx, mn = max(vs), min(vs)
        if mn > 1 and mx/(mn+1e-6) > 20:
            points += 18; reasons.append(f"asimetrik keskinlik (oran={mx/mn:.1f})")

    if w >= 30:
        half = w//2
        lh = arr[:, :half]; rf = arr[:, half:half*2][:, ::-1]
        mn = min(lh.shape[1], rf.shape[1])
        sim = 1.0 - np.abs(lh[:,:mn]-rf[:,:mn]).mean()/255
        if sim > 0.995:
            points += 28; reasons.append(f"mükemmel yatay simetri ({sim:.4f})")

    return {"confidence": min(points / 100.0, 1.0), "reasons": reasons}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cnn(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model   = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, preprocess


@torch.no_grad()
def extract_rotation_bank(model, preprocess, device, img: Image.Image) -> torch.Tensor:
    vecs = []
    for transform in ROTATION_TRANSFORMS:
        rotated = transform(img)
        t = preprocess(rotated).unsqueeze(0).to(device)
        vec = model(t).squeeze(0).cpu()
        vecs.append(vec)
    bank = torch.stack(vecs)
    return nn.functional.normalize(bank, p=2, dim=1)


def rotation_invariant_cosine(bank_i: torch.Tensor, bank_j: torch.Tensor) -> float:
    sim_matrix = torch.mm(bank_i, bank_j.T)
    return sim_matrix.max().item()


def compute_rotation_phashes(img: Image.Image):
    if not IMAGEHASH_OK:
        return None
    hashes = []
    try:
        for transform in ROTATION_TRANSFORMS:
            rotated = transform(img)
            ph = imagehash.phash(rotated, hash_size=16)
            dh = imagehash.dhash(rotated, hash_size=16)
            hashes.append((ph, dh))
    except Exception:
        return None
    return hashes


def rotation_invariant_phash_distance(hashes_i, hashes_j) -> int:
    if hashes_i is None or hashes_j is None:
        return 9999
    min_dist = 9999
    for phi, dhi in hashes_i:
        for phj, dhj in hashes_j:
            dist = ((phi - phj) + (dhi - dhj)) // 2
            if dist < min_dist:
                min_dist = dist
    return min_dist


# ═════════════════════════════════════════════════════════════════════════════
# ANA TEMİZLEYİCİ
# ═════════════════════════════════════════════════════════════════════════════

class DatasetCleaner:
    IMG_EXTS = {"*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.bmp","*.BMP"}

    def __init__(
        self,
        root_dir: str,
        output_dir: str,
        removed_dir: str = None,
        cnn_sim_threshold: float  = 0.82,
        phash_threshold:  int     = 14,
        aug_score_threshold: float = 0.42,
    ):
        self.root    = Path(root_dir).resolve()
        self.out     = Path(output_dir).resolve()
        self.removed = (Path(removed_dir).resolve() if removed_dir
                        else self.out.parent / "kaldirilan_auglar")
        self.cnn_thr   = cnn_sim_threshold
        self.phash_thr = phash_threshold
        self.aug_thr   = aug_score_threshold
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Cihaz: {'CUDA ✓' if torch.cuda.is_available() else 'CPU'}")
        if not IMAGEHASH_OK:
            log.warning("imagehash yüklü değil → pHash aşaması atlanıyor!")

        self._report = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.2-all-angles-invariant",
            "settings": {
                "cnn_sim_threshold": cnn_sim_threshold,
                "phash_threshold":   phash_threshold,
                "aug_score_threshold": aug_score_threshold,
                "rotation_variants":  N_ROTS,
            },
            "classes": {},
        }

    def _scan(self, folder: Path) -> list:
        imgs = []
        for ext in self.IMG_EXTS:
            imgs.extend(folder.rglob(ext))
        return sorted(set(imgs))

    def _load(self, path: Path):
        try:
            if path.stat().st_size == 0:
                return None
            with Image.open(path) as chk: chk.verify()
            img = Image.open(path)
            return img.convert("RGB") if img.mode != "RGB" else img
        except Exception as e:
            log.debug(f"Atlandı: {path.name} → {e}")
            return None

    def _originality(self, path: Path, img: Image.Image):
        sf = filename_aug_score(path)
        se = exif_aug_score(path, img)
        sp = pixel_artifact_score(img)
        aug_prob = 0.28*sf["confidence"] + 0.32*se["confidence"] + 0.40*sp["confidence"]
        return 1.0 - aug_prob, {"filename": sf, "exif": se, "pixel": sp}

    def _union_find(self, n: int, pairs: list) -> list:
        parent = list(range(n))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py: parent[px] = py
        for i, j in pairs: union(i, j)
        groups = defaultdict(list)
        for i in range(n): groups[find(i)].append(i)
        return list(groups.values())

    def _process_folder(self, folder: Path, model, preprocess):
        paths = self._scan(folder)
        if not paths:
            return
        rel = folder.relative_to(self.root)
        log.info(f"\n{'─'*62}")
        log.info(f"📂  {rel}  ({len(paths)} görsel)")

        records = []
        for p in tqdm(paths, desc="  Ön analiz", leave=False):
            img = self._load(p)
            if img is None: continue
            orig, stages = self._originality(p, img)
            records.append({
                "path": p, "img": img,
                "originality": orig, "stages": stages,
                "md5": md5_of(p),
            })

        if not records:
            log.warning(f"  Geçerli görsel yok: {rel}")
            return

        n = len(records)

        log.info(f"  Rotasyon bankası oluşturuluyor ({n} görsel × {N_ROTS} açı)...")
        for r in tqdm(records, desc="  CNN banka", leave=False):
            r["rot_bank"] = extract_rotation_bank(model, preprocess, self.device, r["img"])

        if IMAGEHASH_OK:
            for r in tqdm(records, desc="  pHash", leave=False):
                r["rot_hashes"] = compute_rotation_phashes(r["img"])
        else:
            for r in records:
                r["rot_hashes"] = None

        similar_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                similar = False
                why     = []

                if records[i]["md5"] == records[j]["md5"]:
                    similar = True
                    why.append("md5_exact")

                if not similar:
                    pd = rotation_invariant_phash_distance(
                        records[i]["rot_hashes"], records[j]["rot_hashes"]
                    )
                    if pd <= self.phash_thr:
                        similar = True
                        why.append(f"phash_rot_dist={pd}")

                if not similar:
                    cnn_sim = rotation_invariant_cosine(
                        records[i]["rot_bank"], records[j]["rot_bank"]
                    )
                    if cnn_sim >= self.cnn_thr:
                        similar = True
                        why.append(f"cnn_rot_max={cnn_sim:.3f}")

                if similar:
                    similar_pairs.append((i, j))
                    log.debug(
                        f"    Benzer: {records[i]['path'].name} ↔ "
                        f"{records[j]['path'].name}  [{', '.join(why)}]"
                    )

        clusters = self._union_find(n, similar_pairs)

        dest = self.out / rel
        dest.mkdir(parents=True, exist_ok=True)
        rem_base = self.removed / rel

        kept, removed = [], []
        suspicious = []

        for cidx in clusters:
            cluster = [records[i] for i in cidx]

            if len(cluster) == 1:
                r = cluster[0]
                shutil.copy2(r["path"], dest / r["path"].name)
                entry = {"file": r["path"].name, "originality": round(r["originality"], 3)}
                if r["originality"] < (1.0 - self.aug_thr):
                    entry["warning"] = "tekil ama aug skoru yüksek"
                    suspicious.append(r["path"].name)
                kept.append(entry)
                continue

            cluster.sort(key=lambda x: x["originality"], reverse=True)
            best = cluster[0]
            rest = cluster[1:]

            shutil.copy2(best["path"], dest / best["path"].name)
            kept.append({
                "file":         best["path"].name,
                "originality":  round(best["originality"], 3),
                "cluster_size": len(cluster),
                "beat_out":     [x["path"].name for x in rest[:6]],
            })

            rem_base.mkdir(parents=True, exist_ok=True)
            for r in rest:
                shutil.copy2(r["path"], rem_base / r["path"].name)
                removed.append({
                    "file":         r["path"].name,
                    "originality":  round(r["originality"], 3),
                    "kept_instead": best["path"].name,
                    "aug_confidence": {
                        k: round(v["confidence"], 3) for k, v in r["stages"].items()
                    },
                    "aug_reasons": {
                        k: v["reasons"] for k, v in r["stages"].items() if v["reasons"]
                    },
                })

        cls_key = str(rel)
        self._report["classes"][cls_key] = {
            "input_count": len(paths),
            "valid_count": n,
            "kept":        len(kept),
            "removed":     len(removed),
            "suspicious_singletons": suspicious,
            "kept_files":  kept,
            "removed_files": removed,
        }

        log.info(
            f"  ✓ {n} geçerli → {len(kept)} korundu, "
            f"{len(removed)} kaldırıldı"
            + (f", {len(suspicious)} şüpheli" if suspicious else "")
        )

    def run(self):
        log.info("=" * 62)
        log.info(f"  DATASET TEMİZLEYİCİ v2.2 — HER AÇIDAN ROTASYON ({STEP}°) ")
        log.info("=" * 62)

        model, preprocess = build_cnn(self.device)
        log.info("ResNet18 hazır.")

        all_folders = set()
        for ext in self.IMG_EXTS:
            for p in self.root.rglob(ext):
                all_folders.add(p.parent)

        if not all_folders:
            log.error(f"Görsel bulunamadı: {self.root}")
            return

        log.info(f"{len(all_folders)} klasör bulundu.")

        for folder in sorted(all_folders):
            self._process_folder(folder, model, preprocess)

        ti = sum(d["input_count"] for d in self._report["classes"].values())
        tk = sum(d["kept"]        for d in self._report["classes"].values())
        tr = sum(d["removed"]     for d in self._report["classes"].values())
        self._report["totals"] = {
            "total_input":   ti,
            "total_kept":    tk,
            "total_removed": tr,
            "reduction_pct": round(tr / max(ti,1) * 100, 1),
        }

        self.out.mkdir(parents=True, exist_ok=True)
        json_path = self.out / "temizlik_raporu_v22.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, ensure_ascii=False, indent=2)

        txt_path = self.out / "temizlik_raporu_v22.txt"
        self._write_txt_report(txt_path, ti, tk, tr)

        log.info("\n" + "=" * 62)
        log.info(f"  TAMAMLANDI")
        log.info(f"  Toplam girdi   : {ti}")
        log.info(f"  Korunan        : {tk}")
        log.info(f"  Kaldırılan     : {tr}  (%{self._report['totals']['reduction_pct']})")
        log.info(f"  JSON raporu    : {json_path}")
        log.info(f"  TXT raporu     : {txt_path}")
        log.info(f"  Kaldırılanlar  : {self.removed}")
        log.info("=" * 62)

    def _write_txt_report(self, path: Path, ti, tk, tr):
        lines = [
            f"DATASET TEMİZLEME RAPORU  v2.2 — HER AÇIDAN ROTASYON ({STEP}°)",
            f"Tarih: {self._report['timestamp']}",
            f"Rotasyon varyantları: {N_ROTS}",
            f"CNN eşiği: {self.cnn_thr}  |  pHash eşiği: {self.phash_thr}",
            "─" * 60,
        ]
        for cls_key, data in self._report["classes"].items():
            lines.append(f"\n[{cls_key}]")
            lines.append(
                f"  Girdi: {data['input_count']}  |  Korunan: {data['kept']}  |  "
                f"Kaldırılan: {data['removed']}"
            )
            for r in data["removed_files"][:20]:
                sc = " │ ".join(f"{k[:4]}:{v:.0%}" for k,v in r["aug_confidence"].items())
                lines.append(f"    ✗ {r['file']:<40} [{sc}]  → {r['kept_instead']}")
            if len(data["removed_files"]) > 20:
                lines.append(f"    ... +{len(data['removed_files'])-20} daha (tam liste JSON'da)")
        lines += [
            "", "═"*60,
            f"TOPLAM: {ti} girdi  →  {tk} korundu,  {tr} kaldırıldı  "
            f"(%{self._report['totals']['reduction_pct']} azalma)",
            "", f"Kaldırılanlar silinmedi → {self.removed}",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# AYARLAR
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    cleaner = DatasetCleaner(
        root_dir    = r"C:\Users\kemal\PycharmProjects\cnn\dataset_animals",
        output_dir  = r"C:\Users\kemal\PycharmProjects\cnn\temiz_veri_v22",
        removed_dir = r"C:\Users\kemal\PycharmProjects\cnn\kaldirilan_auglar_v22",

        cnn_sim_threshold = 0.94,
        phash_threshold = 8,
        aug_score_threshold = 0.42,
    )

    cleaner.run()