from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\Users\LENOVO\PhoneDetector")
MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "detect" / "phone_finetune-2" / "weights" / "best.pt"
DATA_PATH = ROOT / "custom_dataset" / "phone_finetune_release_01" / "data.yaml"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Veri YAML dosyası bulunamadı: {DATA_PATH}")

model = YOLO(str(MODEL_PATH))

results = model.train(
    data=str(DATA_PATH),
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project=str(ROOT / "runs" / "detect"),
    name="phone_finetune_release_01",
    exist_ok=False,
    patience=5,
    plots=False,
    seed=42
)

print("\nEĞİTİM TAMAMLANDI")
print("Sonuç klasörü:", results.save_dir)
print("En iyi ağırlık dosyası:", Path(results.save_dir) / "weights" / "best.pt")
