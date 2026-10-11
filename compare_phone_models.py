from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.files import increment_path


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "Cell-Phone-Dataset-detect" / "data.yaml"
ORIGINAL_DATA = ROOT / "Cell-Phone-Dataset-detect"
BASELINE = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "phone_finetune-2"
    / "weights"
    / "best.pt"
)
UPDATED = (
    ROOT
    / "runs"
    / "detect"
    / "phone_finetune_release_02"
    / "cpu_finetune_03"
    / "weights"
    / "best.pt"
)
OUTPUT = ROOT / "runs" / "detect" / "phone_finetune_comparison"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHONE_CLASS_ID = 1
PREDICTION_CONFIDENCE = 0.25
MATCH_IOU = 0.5


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    area_second = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def ground_truth_boxes(image: Path) -> list[np.ndarray]:
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    boxes: list[np.ndarray] = []
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for evaluation image: {label}")
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        values = [float(value) for value in fields]
        class_id, x_center, y_center, width, height = values
        if int(class_id) == PHONE_CLASS_ID:
            boxes.append(
                np.array(
                    [
                        x_center - width / 2,
                        y_center - height / 2,
                        x_center + width / 2,
                        y_center + height / 2,
                    ],
                    dtype=np.float32,
                )
            )
    unique = {tuple(float(value) for value in row) for row in boxes}
    return [np.asarray(row, dtype=np.float32) for row in unique]


def count_phone_errors(model: YOLO, split: str) -> dict[str, Any]:
    image_dir = ORIGINAL_DATA / split / "images"
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    false_positives: list[str] = []
    missed_phones: list[str] = []
    fp_boxes = 0
    fn_boxes = 0
    processed = 0

    results = model.predict(
        source=[str(image) for image in images],
        stream=True,
        imgsz=416,
        batch=4,
        device="cpu",
        classes=[PHONE_CLASS_ID],
        conf=PREDICTION_CONFIDENCE,
        iou=0.45,
        verbose=False,
    )
    for image, result in zip(images, results):
        processed += 1
        height, width = result.orig_shape
        truths = ground_truth_boxes(image)
        predictions = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None
            else np.empty((0, 4), dtype=np.float32)
        )
        predictions = predictions / np.array(
            [width, height, width, height], dtype=np.float32
        )
        candidates = sorted(
            (
                (box_iou(truth, prediction), truth_index, prediction_index)
                for truth_index, truth in enumerate(truths)
                for prediction_index, prediction in enumerate(predictions)
            ),
            reverse=True,
        )
        matched_truths: set[int] = set()
        matched_predictions: set[int] = set()
        for overlap, truth_index, prediction_index in candidates:
            if overlap < MATCH_IOU:
                break
            if (
                truth_index not in matched_truths
                and prediction_index not in matched_predictions
            ):
                matched_truths.add(truth_index)
                matched_predictions.add(prediction_index)

        missed = len(truths) - len(matched_truths)
        extra = len(predictions) - len(matched_predictions)
        if missed:
            missed_phones.append(image.name)
            fn_boxes += missed
        if extra:
            false_positives.append(image.name)
            fp_boxes += extra

    if processed != len(images):
        raise RuntimeError(
            f"{split}: prediction stream returned {processed} of {len(images)} images."
        )
    return {
        "prediction_confidence": PREDICTION_CONFIDENCE,
        "match_iou": MATCH_IOU,
        "false_positive_images": false_positives,
        "false_positive_boxes": fp_boxes,
        "missed_phone_images": missed_phones,
        "missed_phone_boxes": fn_boxes,
    }


def scalar_metric(values: Any, class_indices: Any, class_id: int) -> float:
    indices = [int(value) for value in class_indices]
    if class_id not in indices:
        return 0.0
    return float(values[indices.index(class_id)])


def evaluate_model(model_path: Path, model_key: str) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights are missing: {model_path}")
    model = YOLO(str(model_path), task="detect")
    expected_name = model.names.get(PHONE_CLASS_ID)
    if expected_name != "cell phone":
        raise ValueError(f"{model_path} has unexpected class mapping: {model.names}")

    report: dict[str, Any] = {"weights": str(model_path), "splits": {}}
    for split, split_key in (("valid", "val"), ("test", "test")):
        run_path = increment_path(
            OUTPUT / f"{model_key}_{split}", exist_ok=False, mkdir=False
        )
        results = model.val(
            data=str(DATA_YAML),
            split=split_key,
            imgsz=416,
            batch=2,
            device="cpu",
            workers=0,
            plots=False,
            project=str(run_path.parent),
            name=run_path.name,
            exist_ok=False,
            verbose=False,
        )
        metrics = results.box
        class_indices = metrics.ap_class_index
        per_class = {
            "precision": scalar_metric(metrics.p, class_indices, PHONE_CLASS_ID),
            "recall": scalar_metric(metrics.r, class_indices, PHONE_CLASS_ID),
            "mAP50": scalar_metric(metrics.ap50, class_indices, PHONE_CLASS_ID),
            "mAP50_95": scalar_metric(metrics.ap, class_indices, PHONE_CLASS_ID),
        }
        report["splits"][split] = {
            "overall": {
                "precision": float(metrics.mp),
                "recall": float(metrics.mr),
                "mAP50": float(metrics.map50),
                "mAP50_95": float(metrics.map),
            },
            "cell_phone": per_class,
            "false_positives_and_misses": count_phone_errors(model, split),
            "validation_run": str(results.save_dir),
        }
        print(
            f"{model_key} {split}: cell phone={per_class}, "
            f"overall={report['splits'][split]['overall']}"
        )

    del model
    gc.collect()
    return report


def compare() -> Path:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Evaluation dataset is missing: {DATA_YAML}")
    baseline = evaluate_model(BASELINE, "baseline")
    updated = evaluate_model(UPDATED, "updated")
    comparison = {
        "evaluation_dataset": str(DATA_YAML),
        "same_original_validation_and_test_images_for_both_models": True,
        "phone_error_operating_point": {
            "confidence": PREDICTION_CONFIDENCE,
            "IoU_match": MATCH_IOU,
        },
        "baseline": baseline,
        "updated": updated,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "release_02_comparison.json"
    report_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Comparison report: {report_path}")
    return report_path


if __name__ == "__main__":
    compare()
