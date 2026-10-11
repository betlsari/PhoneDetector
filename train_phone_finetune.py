from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from validate_phone_finetune_dataset import validate_dataset


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "custom_dataset" / "phone_finetune_release_02"
MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "phone_finetune-2"
    / "weights"
    / "best.pt"
)
RUNS = ROOT / "runs" / "detect" / "phone_finetune_release_02"


def train(*, trial: bool) -> Path:
    data_path = DATASET / "data.yaml"
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Starting weights are missing: {MODEL_PATH}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Validated dataset is missing: {data_path}")
    validate_dataset(DATASET)

    name = "cpu_trial_01" if trial else "cpu_finetune_03"
    model = YOLO(str(MODEL_PATH), task="detect")
    results = model.train(
        data=str(data_path),
        epochs=1 if trial else 10,
        imgsz=320 if trial else 320,
        batch=1,
        device="cpu",
        workers=0,
        project=str(RUNS),
        name=name,
        exist_ok=False,
        patience=1 if trial else 4,
        plots=False,
        cache=False,
        seed=42,
        deterministic=True,
        fraction=0.02 if trial else 1.0,
        amp=False,
        close_mosaic=0 if trial else 3,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training ended without best weights: {best}")
    print(f"Training complete: {results.save_dir}")
    print(f"Best weights: {best}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a CPU smoke trial or full phone fine-tuning."
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run one epoch on 2% of training images to validate configuration.",
    )
    train(trial=parser.parse_args().trial)
