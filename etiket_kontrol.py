from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_DIR / "custom_dataset" / "new_photos" / "phone"
LABEL_DIR = PROJECT_DIR / "custom_dataset" / "phone_finetune_release_03_corrected" / "labels" / "train"
OUTPUT_DIR = PROJECT_DIR / "custom_dataset" / "label_review_corrected"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "WhatsApp Image 2026-10-09 at 19.58.06 (1).jpeg",
    "WhatsApp Image 2026-10-09 at 19.58.11.jpeg",
]

for filename in files:
    image_path = IMAGE_DIR / filename
    label_path = LABEL_DIR / (Path(filename).stem + ".txt")

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Fotoğraf okunamadı: {image_path}")
        continue

    if not label_path.exists():
        print(f"Etiket dosyası bulunamadı: {label_path}")
        continue

    height, width = image.shape[:2]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id, xc, yc, bw, bh = parts
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image,
            f"class {class_id}",
            (max(0, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    print(f"Kaydedildi: {output_path}")

print("Etiket kontrolü tamamlandı.")