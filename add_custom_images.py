from pathlib import Path
import shutil

src_images = Path("custom_dataset/roboflow_export/train/images")
src_labels = Path("custom_dataset/labels_boxes")

dst_images = Path("Cell-Phone-Dataset/train/images")
dst_labels = Path("Cell-Phone-Dataset/train/labels")

dst_images.mkdir(parents=True, exist_ok=True)
dst_labels.mkdir(parents=True, exist_ok=True)

for image in src_images.glob("*.jpg"):
    label = src_labels / (image.stem + ".txt")

    if not label.exists():
        print(f"Etiket bulunamadı: {image.name}")
        continue

    shutil.copy2(image, dst_images / image.name)
    shutil.copy2(label, dst_labels / label.name)

print("Yeni fotoğraflar ve etiketler eğitim klasörüne kopyalandı.")
