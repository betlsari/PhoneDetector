from pathlib import Path
import shutil

source = Path("custom_dataset/images/train")
target_images = Path("Cell-Phone-Dataset-detect/train/images")
target_labels = Path("Cell-Phone-Dataset-detect/train/labels")

files = sorted(
    p for p in source.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
)

selected_numbers = [2, 3, 6, 11]

target_images.mkdir(parents=True, exist_ok=True)
target_labels.mkdir(parents=True, exist_ok=True)

for number in selected_numbers:
    if number > len(files):
        raise ValueError(f"{number}. fotoğraf bulunamadı.")

    image = files[number - 1]
    destination = target_images / image.name
    label = target_labels / f"{image.stem}.txt"

    if destination.exists():
        print(f"Zaten var, atlandı: {image.name}")
        continue

    shutil.copy2(image, destination)
    label.write_text("", encoding="utf-8")
    print(f"Eklendi: {image.name} (boş etiket)")

print("\nBoş kılıf örnekleri işleme alındı.")
