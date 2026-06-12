import os

# =============================================================
# ⚠️ KENDİ BİLGİSAYARINDAKİ KLASÖR YOLUNU YAZ
# Örnek: "C:/Kullanicilar/Masaustu/hayvan_projesi/dataset_animals"
# =============================================================
BASE_DIR = r'C:\Users\kemal\PycharmProjects\cnn\temiz_veri_v22\dataset_animals'

if not os.path.exists(BASE_DIR):
    print(f"Hata: Klasör bulunamadı: {BASE_DIR}")
else:
    print(f"Klasör işleniyor: {BASE_DIR}\n")

    for class_folder in os.listdir(BASE_DIR):
        class_path = os.path.join(BASE_DIR, class_folder)

        if not os.path.isdir(class_path):
            continue

        print(f"Sınıf: {class_folder} isimlendiriliyor...")

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()

        for index, img_name in enumerate(images, start=1):
            extension = os.path.splitext(img_name)[1].lower()
            new_name = f"{class_folder}_{index:03d}{extension}"

            old_file_path = os.path.join(class_path, img_name)
            new_file_path = os.path.join(class_path, new_name)

            if old_file_path == new_file_path:
                continue

            try:
                os.rename(old_file_path, new_file_path)
            except Exception as e:
                print(f"  Hata: {img_name} -> {new_name} yapılamadı. {e}")

    print("\n✅ Tamamlandı! Tüm fotoğraflar formatına uygun şekilde isimlendirildi.")
