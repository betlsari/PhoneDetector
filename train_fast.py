import random
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\Users\LENOVO\PhoneDetector")
RELEASE = ROOT / "custom_dataset" / "phone_finetune_release_01"
MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "detect" / "phone_finetune-2" / "weights" / "best.pt"
ORIGINAL_TRAIN = ROOT / "Cell-Phone-Dataset-detect" / "train" / "images"
VALID = ROOT / "Cell-Phone-Dataset-detect" / "valid" / "images"
TEST = ROOT / "Cell-Phone-Dataset-detect" / "test" / "images"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

if not (RELEASE / "data.yaml").is_file():
    raise FileNotFoundError("Hazırlanan veri seti bulunamadı.")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

originals = [
    p for p in ORIGINAL_TRAIN.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]
custom = [
    p for p in (RELEASE / "images" / "train").iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

if len(originals) < 300 or len(custom) != 21:
    raise ValueError(
        f"Beklenmeyen görsel sayısı: orijinal={len(originals)}, yeni={len(custom)}"
    )

random.seed(42)
selected = random.sample(originals, 300) + custom

fast_list = RELEASE / "train_fast.txt"
fast_list.write_text(
    "\n".join(p.resolve().as_posix() for p in selected) + "\n",
    encoding="utf-8"
)

yaml_text = f"""path: {ROOT.as_posix()}
train: {fast_list.resolve().as_posix()}
val: {VALID.resolve().as_posix()}
test: {TEST.resolve().as_posix()}

nc: 2
names: ['0', 'cell phone']
"""
fast_yaml = RELEASE / "data_fast.yaml"
fast_yaml.write_text(yaml_text, encoding="utf-8")

print("Hızlı eğitim görselleri:", len(selected))
print("Yeni fotoğraflar:", len(custom))
print("YAML:", fast_yaml)

model = YOLO(str(MODEL_PATH))
results = model.train(
    data=str(fast_yaml),
    epochs=3,
    imgsz=416,
    batch=8,
    device="cpu",
    workers=0,
    project=str(ROOT / "runs" / "detect"),
    name="phone_finetune_fast_01",
    exist_ok=False,
    patience=2,
    plots=False,
    cache=False,
    seed=42
)

print("\nHIZLI EĞİTİM TAMAMLANDI")
print("Sonuç klasörü:", results.save_dir)
print("En iyi model:", Path(results.save_dir) / "weights" / "best.pt")
