import os
from PIL import Image
import imagehash
from pathlib import Path
import shutil

def find_original_images(root_dir, output_dir):
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        print(f"HATA: '{root_path}' dizini bulunamadı!")
        return

    print(f"Tarama başlıyor: {root_path}")
    disease_folders = [d for d in root_path.iterdir() if d.is_dir()]

    for disease_path in disease_folders:
        print(f"İşleniyor: {disease_path.name}")
        hashes = {}
        image_paths = list(disease_path.glob('*.[jJ][pP][gG]')) + \
                      list(disease_path.glob('*.[pP][nN][gG]')) + \
                      list(disease_path.glob('*.[jJ][pP][eE][gG]'))

        for img_path in image_paths:
            try:
                # 1. Adım: Dosya boyutu 0 ise doğrudan atla
                if img_path.stat().st_size == 0:
                    continue

                # 2. Adım: Görseli aç ve doğrula
                with Image.open(img_path) as img:
                    img.verify() # Dosya içeriğinin bozuk olup olmadığını kontrol eder

                # verify() sonrası dosyayı tekrar açmamız gerekir çünkü verify dosyayı kapatır
                with Image.open(img_path) as img:
                    h = str(imagehash.phash(img))
                    has_exif = bool(img._getexif()) if hasattr(img, '_getexif') else False
                    file_size = img_path.stat().st_size

                    if h not in hashes:
                        hashes[h] = []

                    hashes[h].append({
                        'path': img_path,
                        'has_exif': has_exif,
                        'size': file_size,
                        'name': img_path.name
                    })
            except Exception:
                # Bozuk dosya olduğunda hiçbir şey yapmadan bir sonrakine geçer
                # Konsolda kalabalık yapmaması için print'i kaldırdım, sadece atlar.
                continue

        # Orijinalleri seç ve kopyala
        for h, files in hashes.items():
            if not files: continue # Eğer bu grupta hiç sağlam dosya kalmadıysa atla

            best_match = sorted(files,
                                key=lambda x: (x['has_exif'], -len(x['name']), x['size']),
                                reverse=True)[0]

            dest_dir = Path(output_dir) / disease_path.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_match['path'], dest_dir / best_match['name'])

    print(f"\nİşlem tamamlandı! Temiz veriler şurada: {output_dir}")

# --- AYARLAR ---
dataset_path = r'C:\Users\kemal\PycharmProjects\cnn\köpekvetavukveriseti\poultry'
target_path = r'C:\Users\kemal\PycharmProjects\cnn\temizlenmis_poultry_verisi_serhanlar'

if __name__ == "__main__":
    find_original_images(dataset_path, target_path)