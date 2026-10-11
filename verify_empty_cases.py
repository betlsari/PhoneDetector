from pathlib import Path

images = Path("Cell-Phone-Dataset-detect/train/images")
labels = Path("Cell-Phone-Dataset-detect/train/labels")

empty = []
for image in images.iterdir():
    if image.name.startswith("WhatsApp Image 2026-10-09"):
        label = labels / f"{image.stem}.txt"
        if label.exists() and label.stat().st_size == 0:
            empty.append(image.name)

print("Boş etiketli WhatsApp fotoğrafları:", len(empty))
for name in empty:
    print("-", name)
