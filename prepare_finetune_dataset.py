import json
import hashlib
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\LENOVO\PhoneDetector")
JSON_FILE = Path.home() / "Downloads" / "phone_box_review_approved.json"
PHONE_DIR = ROOT / "custom_dataset" / "new_photos" / "phone"
EMPTY_DIR = ROOT / "custom_dataset" / "new_photos" / "empty_case"
ORIGINAL_TRAIN = ROOT / "Cell-Phone-Dataset-detect" / "train" / "images"
ORIGINAL_VALID = ROOT / "Cell-Phone-Dataset-detect" / "valid" / "images"
ORIGINAL_TEST = ROOT / "Cell-Phone-Dataset-detect" / "test" / "images"
OUT = ROOT / "custom_dataset" / "phone_finetune_release_01"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_labels(item):
    raw = item.get("yolo_labels")
    labels = []

    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                labels.extend(x.strip() for x in entry.splitlines() if x.strip())
            elif isinstance(entry, (list, tuple)):
                labels.append(" ".join(map(str, entry)))

    # If labels are absent, build them from the reviewed boxes.
    if not labels:
        for box in item.get("boxes", []):
            cid = int(box.get("class_id", 1))
            xc = box.get("x_center", box.get("cx"))
            yc = box.get("y_center", box.get("cy"))
            w = box.get("width", box.get("w"))
            h = box.get("height", box.get("h"))
            if None in (xc, yc, w, h):
                raise ValueError(f"Kutunun koordinat alanları tanınmadı: {box}")
            labels.append(f"{cid} {xc} {yc} {w} {h}")

    if not labels:
        raise ValueError(f"Etiket bulunamadı: {item}")

    for line in labels:
        p = line.split()
        if len(p) != 5:
            raise ValueError(f"Geçersiz YOLO etiketi: {line}")
        cid = int(p[0])
        vals = [float(x) for x in p[1:]]
        if cid not in (0, 1):
            raise ValueError(f"Beklenmeyen sınıf numarası: {line}")
        if not all(0 <= v <= 1 for v in vals):
            raise ValueError(f"Koordinat 0-1 dışında: {line}")
        if vals[2] <= 0 or vals[3] <= 0:
            raise ValueError(f"Kutu boyutu pozitif değil: {line}")
        if cid != 1:
            raise ValueError(f"Yeni telefon etiketi sınıf 1 olmalı: {line}")

    return labels

for p in (JSON_FILE, PHONE_DIR, EMPTY_DIR, ORIGINAL_TRAIN,
          ORIGINAL_VALID, ORIGINAL_TEST):
    if not p.exists():
        raise FileNotFoundError(f"Gerekli dosya/klasör bulunamadı: {p}")

if OUT.exists():
    raise FileExistsError(
        f"{OUT} zaten var. Hiçbir dosyayı ezmedim. "
        "Yeni bir klasör adı seç veya önce içeriğini kontrol et."
    )

data = json.loads(JSON_FILE.read_text(encoding="utf-8"))
reviewed = data.get("reviewed", [])
if len(reviewed) != 18:
    raise ValueError(f"18 inceleme bekleniyordu; bulunan: {len(reviewed)}")

original_images = sorted(
    p for p in ORIGINAL_TRAIN.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)
if not original_images:
    raise ValueError("Orijinal eğitim görselleri bulunamadı.")

# Detect exact duplicates so the same image is not added twice.
known_hashes = {sha256(p) for p in original_images}

# Validate all reviewed photos and labels before creating output.
approved = []
for item in reviewed:
    if item.get("decision") != "correct":
        raise ValueError(f"İnceleme kararı correct değil: {item}")

    name = Path(item.get("image", item.get("image_path", ""))).name
    if not name:
        raise ValueError(f"JSON'da fotoğraf adı bulunamadı: {item}")

    source = PHONE_DIR / name
    if not source.is_file():
        raise FileNotFoundError(f"JSON fotoğrafı bulunamadı: {source}")

    labels = read_labels(item)
    approved.append((source, labels))

OUT.mkdir(parents=True)
images_out = OUT / "images" / "train"
labels_out = OUT / "labels" / "train"
images_out.mkdir(parents=True)
labels_out.mkdir(parents=True)

# Start with all original training images.
train_list = [p.resolve().as_posix() for p in original_images]
added_phone = 0
skipped_duplicates = 0

for source, labels in approved:
    digest = sha256(source)
    if digest in known_hashes:
        skipped_duplicates += 1
        continue

    index = added_phone + 1
    dest_name = f"custom_phone_{index:03d}{source.suffix.lower()}"
    dest_image = images_out / dest_name
    dest_label = labels_out / f"{Path(dest_name).stem}.txt"

    shutil.copy2(source, dest_image)
    dest_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
    train_list.append(dest_image.resolve().as_posix())
    known_hashes.add(digest)
    added_phone += 1

# Add empty cases as negative examples (empty label files).
added_empty = 0
for source in sorted(EMPTY_DIR.iterdir()):
    if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
        continue

    digest = sha256(source)
    if digest in known_hashes:
        skipped_duplicates += 1
        continue

    index = added_empty + 1
    dest_name = f"empty_case_{index:03d}{source.suffix.lower()}"
    dest_image = images_out / dest_name
    dest_label = labels_out / f"{Path(dest_name).stem}.txt"

    shutil.copy2(source, dest_image)
    dest_label.write_text("", encoding="utf-8")
    train_list.append(dest_image.resolve().as_posix())
    known_hashes.add(digest)
    added_empty += 1

train_txt = OUT / "train.txt"
train_txt.write_text("\n".join(train_list) + "\n", encoding="utf-8")

yaml_text = f"""path: {ROOT.as_posix()}
train: {train_txt.resolve().as_posix()}
val: {ORIGINAL_VALID.resolve().as_posix()}
test: {ORIGINAL_TEST.resolve().as_posix()}

nc: 2
names: ['0', 'cell phone']
"""
(OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

# Final sanity checks.
if len(train_list) != len(original_images) + added_phone + added_empty:
    raise RuntimeError("Eğitim listesi sayısı tutarsız.")

for line in train_list:
    if not Path(line).is_file():
        raise FileNotFoundError(f"Eğitim listesinde eksik görsel: {line}")

for image in images_out.iterdir():
    if image.is_file():
        label = labels_out / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Yeni görselin etiketi eksik: {label}")

print("\nHAZIRLIK TAMAMLANDI")
print("Yeni veri seti:", OUT)
print("Eski eğitim görseli:", len(original_images))
print("Eklenen telefon:", added_phone)
print("Eklenen boş kılıf:", added_empty)
print("Tekrarlı görsel olarak atlanan:", skipped_duplicates)
print("Toplam eğitim listesi:", len(train_list))
print("Yeni YAML:", OUT / "data.yaml")
print("Orijinal veri seti değiştirilmedi.")
print("Henüz eğitim başlatılmadı.")
