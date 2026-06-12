import os
from collections import defaultdict
from PIL import Image
import imagehash

# ---------------- AYARLAR ----------------
VERI_SETI_YOLU = r"C:\Users\kemal\PycharmProjects\cnn\dataset_animals"
BENZERLIK_ESIGI = 12 # Görsel analiz için hassasiyet eşiği (10-15 arası ideal)
# -----------------------------------------

def is_valid_image(filepath):
    return filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def get_text_basename(filename):
    """
    Dosya ismindeki kuralları (özellikle Roboflow .rf. kuralı) yakalar
    ve orijinal kök ismi döndürür.
    """
    # Kural 1: Roboflow formatı (örn: 109_png.rf.83adba238...jpg)
    if '.rf.' in filename:
        return filename.split('.rf.')[0]

    # Kural 2: Noktalı format (örn: csv_2106_healthy.100.jpg)
    # İlk noktadan önceki kısmı kök isim kabul ederiz.
    if '.' in filename:
        return filename.split('.')[0]

    # Kural yoksa ismin kendisini uzantısız döndür
    name, _ = os.path.splitext(filename)
    return name

def analyze_hybrid_balance(dataset_path):
    print(f"{'='*70}\nMELEZ (METİN + GÖRSEL) AUGMENTATION ANALİZ RAPORU\n{'='*70}")

    for root, dirs, files in os.walk(dataset_path):
        image_files = [f for f in files if is_valid_image(f)]

        if not image_files:
            continue

        class_name = os.path.basename(root)
        print(f"\n[{class_name.upper()}] Sınıfı İnceleniyor (Toplam {len(image_files)} Fotoğraf)")
        print("-" * 50)

        # --- 1. METİN ANALİZİ (Dosya İsimlendirme Kuralları) ---
        text_counts = defaultdict(int)
        for img in image_files:
            base_name = get_text_basename(img)
            text_counts[base_name] += 1

        text_sizes = list(text_counts.values())
        text_min = min(text_sizes) if text_sizes else 0
        text_max = max(text_sizes) if text_sizes else 0
        text_balanced = (text_max - text_min) <= 1

        print(f"  [METİN ANALİZİ] Bulunan Kök Dosya: {len(text_counts)} | Çoğaltma Aralığı: {text_min} - {text_max}")
        if text_balanced:
            print("  -> SONUÇ: ✔️ Metin kurallarına göre DÜZENLİ.")
        else:
            print("  -> SONUÇ: ❌ Metin kurallarına göre DENGESİZ!")

        # --- 2. GÖRSEL ANALİZ (Piksel Benzerliği - ImageHash) ---
        hashes = {}
        for img in image_files:
            try:
                with Image.open(os.path.join(root, img)) as opened_img:
                    hashes[img] = imagehash.phash(opened_img)
            except:
                pass # Hatalı dosyaları atla

        clusters = []
        for img, img_hash in hashes.items():
            found_cluster = False
            for cluster in clusters:
                if (img_hash - hashes[cluster[0]]) <= BENZERLIK_ESIGI:
                    cluster.append(img)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([img])

        visual_sizes = [len(c) for c in clusters]
        visual_min = min(visual_sizes) if visual_sizes else 0
        visual_max = max(visual_sizes) if visual_sizes else 0
        visual_balanced = (visual_max - visual_min) <= 1

        print(f"  [GÖRSEL ANALİZ] Bulunan Görsel Küme: {len(clusters)} | Çoğaltma Aralığı: {visual_min} - {visual_max}")
        if visual_balanced:
            print("  -> SONUÇ: ✔️ Görsel benzerliğe göre DÜZENLİ.")
        else:
            print("  -> SONUÇ: ❌ Görsel benzerliğe göre DENGESİZ!")

        # --- 3. ORTAK KARAR VE YORUM ---
        print("  [BİRLİKTE KARAR]: ", end="")
        if text_balanced and visual_balanced:
            print("🌟 Kusursuz! Hem isimlendirme hem de görsel olarak adil çoğaltılmış.")
        elif not text_balanced and not visual_balanced:
            print("🚨 KESİN DENGESİZLİK! Sistemi kuran kod/kişi adil dağıtım yapmamış. Veri seti hatalı çoğaltılmış.")
        elif text_balanced and not visual_balanced:
            print("⚠️ Metne göre düzenli ama görsele göre düzensiz. Muhtemelen augmentasyon işlemleri (döndürme/renk) çok ağır yapılmış ve görsel analiz bazılarını tanıyamamış. Metne (dosya ismine) güvenmek daha doğru olabilir.")
        elif not text_balanced and visual_balanced:
            print("⚠️ Görsele göre düzenli ama metne göre düzensiz. İsimlendirme kurallarında bir hata/karmaşa var, ancak görsel olarak eşit dağılım yapılmış görünüyor.")
        print("=" * 50)

if __name__ == "__main__":
    analyze_hybrid_balance(VERI_SETI_YOLU)