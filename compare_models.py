from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\Users\LENOVO\PhoneDetector")
DATA = ROOT / "Cell-Phone-Dataset-detect" / "data.yaml"

models = {
    "Eski model": ROOT / "runs" / "detect" / "runs" / "detect" / "phone_finetune-2" / "weights" / "best.pt",
    "Yeni model": ROOT / "runs" / "detect" / "phone_finetune_fast_01" / "weights" / "best.pt",
}

for name, path in models.items():
    if not path.is_file():
        print(f"\n{name}: MODEL BULUNAMADI: {path}")
        continue

    print(f"\n{'=' * 50}\n{name}\n{path}")
    model = YOLO(str(path))
    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=416,
        batch=8,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False
    )

    box = metrics.box
    print(f"Precision: {box.mp:.3f}")
    print(f"Recall:    {box.mr:.3f}")
    print(f"mAP50:     {box.map50:.3f}")
    print(f"mAP50-95:  {box.map:.3f}")

    names = model.names
    for i, class_name in names.items():
        if i < len(box.p):
            print(
                f"  Sınıf {i} ({class_name}): "
                f"P={box.p[i]:.3f}, R={box.r[i]:.3f}, "
                f"mAP50={box.ap50[i]:.3f}"
            )
