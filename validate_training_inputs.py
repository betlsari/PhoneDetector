from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import yaml
from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_TEXT_FILES = {"readme.dataset.txt", "readme.roboflow.txt"}
MODEL_RELATIVE_PATH = Path(
    "runs/detect/runs/detect/phone_finetune-2/weights/best.pt"
)


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def class_name_map(names: Any) -> dict[int, str] | None:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError):
            return None
    return None


def is_image_payload(data: bytes) -> bool:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        return True
    except Exception:
        return False


def inspect_label(
    path: Path, names: dict[int, str] | None
) -> tuple[str, list[str], Counter[str], Counter[int]]:
    messages: list[str] = []
    shapes: Counter[str] = Counter()
    classes: Counter[int] = Counter()
    data = path.read_bytes()

    if is_image_payload(data):
        return "INVALID", ["TXT file contains image data"], shapes, classes

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "INVALID", ["not UTF-8 text and not a supported image"], shapes, classes

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            messages.append(f"line {line_number}: non-numeric label value")
            continue

        if class_id < 0:
            messages.append(f"line {line_number}: negative class id {class_id}")
        elif names is not None and class_id not in names:
            messages.append(f"line {line_number}: class id {class_id} is not in data.yaml")
        else:
            classes[class_id] += 1

        if len(coordinates) == 4:
            shape = "box"
            center_x, center_y, width, height = coordinates
            if width <= 0 or height <= 0:
                messages.append(f"line {line_number}: box width/height must be positive")
            if center_x - width / 2 < 0 or center_x + width / 2 > 1:
                messages.append(f"line {line_number}: box exceeds horizontal image bounds")
            if center_y - height / 2 < 0 or center_y + height / 2 > 1:
                messages.append(f"line {line_number}: box exceeds vertical image bounds")
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            shape = "polygon"
        else:
            messages.append(
                f"line {line_number}: expected 4 box values or at least 6 polygon values"
            )
            continue

        if any(not math.isfinite(value) or value < 0 or value > 1 for value in coordinates):
            messages.append(f"line {line_number}: coordinates must be finite and in 0..1")
        shapes[shape] += 1

    if messages:
        return "INVALID", messages, shapes, classes
    return "VALID", messages, shapes, classes


def audit_pairs(
    title: str,
    image_dir: Path,
    label_dir: Path,
    names: dict[int, str] | None,
    *,
    manual_class_review: bool = False,
    verbose: bool = False,
) -> Counter[str]:
    print(f"\n[{title}]")
    if not image_dir.is_dir():
        print(f"MISSING DIRECTORY: {image_dir}")
        return Counter({"MISSING_DIRECTORY": 1})
    if not label_dir.is_dir():
        print(f"MISSING DIRECTORY: {label_dir}")
        return Counter({"MISSING_DIRECTORY": 1})

    images = {path.stem.casefold(): path for path in image_files(image_dir)}
    labels: dict[str, Path] = {}
    ignored: list[str] = []
    for path in sorted(label_dir.glob("*.txt")):
        if path.name.casefold() in IGNORED_TEXT_FILES:
            ignored.append(path.name)
            continue
        labels[path.stem.casefold()] = path

    statuses: Counter[str] = Counter()
    classes: Counter[int] = Counter()
    formats: Counter[str] = Counter()

    for key, image_path in images.items():
        label_path = labels.get(key)
        if label_path is None:
            status = "MISSING_LABEL"
            detail = "no matching .txt label"
        else:
            status, messages, shapes, found_classes = inspect_label(label_path, names)
            classes.update(found_classes)
            formats.update(shapes)
            detail = "; ".join(messages) if messages else ", ".join(
                f"{count} {shape}" for shape, count in sorted(shapes.items())
            ) or "empty label (negative image)"
            if status == "VALID" and manual_class_review:
                status = "MANUAL_CLASS_REVIEW"
                detail += "; class meaning/mapping is not confirmed"
        statuses[status] += 1
        if verbose or status != "VALID":
            print(f"{status}: {image_path.name} -> {detail}")

    for key, label_path in labels.items():
        if key not in images:
            statuses["ORPHAN_LABEL"] += 1
            status, messages, _, _ = inspect_label(label_path, names)
            detail = "; ".join(messages) if messages else status
            print(f"ORPHAN_LABEL: {label_path.name} -> {detail}")

    for name in ignored:
        print(f"IGNORED METADATA (not a label): {name}")
    print(
        "SUMMARY:",
        f"images={len(images)} labels={len(labels)}",
        f"statuses={dict(statuses)}",
        f"classes={dict(classes)} formats={dict(formats)}",
    )
    return statuses


def load_and_audit_yaml(path: Path, *, verbose: bool = False) -> dict[int, str] | None:
    print(f"\n[YAML: {path}]")
    if not path.is_file():
        print("INVALID: file does not exist")
        return None
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as error:
        print(f"INVALID YAML: {error}")
        return None

    if not isinstance(config, dict):
        print(f"INVALID YAML SCHEMA: expected mapping, found {type(config).__name__}")
        print(f"Parsed value: {config!r}")
        return None

    names = class_name_map(config.get("names"))
    nc = config.get("nc")
    if names is None:
        print("INVALID YAML SCHEMA: missing or invalid names")
        return None
    if not isinstance(nc, int) or nc != len(names):
        print(f"INVALID YAML SCHEMA: nc={nc!r}, names count={len(names)}")
        return None
    print(f"VALID YAML: nc={nc}, names={names}")

    dataset_root = Path(config.get("path", path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if split_value is None:
            continue
        split_paths = split_value if isinstance(split_value, list) else [split_value]
        for split_item in split_paths:
            split_path = Path(str(split_item))
            if not split_path.is_absolute():
                split_path = dataset_root / split_path
            if not split_path.is_dir():
                print(f"INVALID PATH: {split} -> {split_path}")
                continue
            audit_pairs(
                f"{path.parent.name}/{split}",
                split_path,
                split_path.parent / "labels",
                names,
                verbose=verbose,
            )
    return names


def audit_exact_duplicates(root: Path) -> None:
    training_images = image_files(
        root / "Cell-Phone-Dataset-detect" / "train" / "images"
    )
    incoming_images = []
    for folder in (
        root / "custom_dataset" / "new_photos" / "phone",
        root / "custom_dataset" / "new_photos" / "empty_case",
        root / "custom_dataset" / "roboflow_new",
    ):
        incoming_images.extend(image_files(folder))

    print("\n[EXACT IMAGE DUPLICATES: SHA-256; read-only]")
    training_hashes: dict[str, Path] = {}
    for path in training_images:
        training_hashes.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)

    incoming_hashes: dict[str, Path] = {}
    duplicates_in_train = 0
    duplicates_in_incoming = 0
    for path in incoming_images:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in training_hashes:
            duplicates_in_train += 1
            print(f"DUPLICATE_OF_TRAIN: {path} == {training_hashes[digest]}")
        elif digest in incoming_hashes:
            duplicates_in_incoming += 1
            print(f"DUPLICATE_IN_INCOMING: {path} == {incoming_hashes[digest]}")
        else:
            incoming_hashes[digest] = path

    print(
        f"SUMMARY: incoming_images={len(incoming_images)} "
        f"duplicates_of_train={duplicates_in_train} "
        f"duplicates_within_incoming={duplicates_in_incoming}"
    )


def allocate_review_directory(requested: Path) -> Path:
    candidate = requested
    index = 1
    while candidate.exists():
        candidate = requested.with_name(f"{requested.name}_{index:02d}")
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def review_phone_predictions(
    root: Path,
    output_base: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
) -> None:
    source = root / "custom_dataset" / "new_photos" / "phone"
    files = image_files(source)
    print(f"\n[NEW PHONE PHOTOS: {source}]")
    print(f"Found {len(files)} images. Model boxes are candidates only, never verified labels.")
    if not files:
        return

    if not model_path.is_file():
        print(f"MODEL MISSING: {model_path}; all photos require manual annotation.")
        for path in files:
            print(f"MANUAL_ANNOTATION_REQUIRED: {path.name}")
        return

    model = YOLO(str(model_path), task="detect")
    names = class_name_map(model.names)
    if names is None or names.get(1, "").casefold() != "cell phone":
        print(f"MODEL CLASS MISMATCH: expected class 1='cell phone', found {model.names!r}")
        for path in files:
            print(f"MANUAL_ANNOTATION_REQUIRED: {path.name}")
        return

    output_dir = allocate_review_directory(output_base)
    csv_path = output_dir / "phone_box_review.csv"
    review_rows: list[dict[str, str]] = []

    for path in files:
        result = model.predict(
            source=str(path),
            imgsz=image_size,
            conf=confidence,
            classes=[1],
            device="cpu",
            verbose=False,
        )[0]
        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                score = float(box.conf[0])
                boxes.append((x_center, y_center, width, height, score))

        status = "MANUAL_REVIEW_BOXES" if boxes else "MANUAL_ANNOTATION_NO_BOX"
        preview = result.plot()
        warning = "UNVERIFIED PREDICTIONS - CHECK BOXES BEFORE USE"
        cv2.putText(
            preview,
            warning,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        encoded, buffer = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if encoded:
            (output_dir / f"{path.stem}_review.jpg").write_bytes(buffer.tobytes())

        if boxes:
            for box_index, (x, y, width, height, score) in enumerate(boxes, start=1):
                review_rows.append(
                    {
                        "image": path.name,
                        "review_status": status,
                        "class_id": "1",
                        "class_name": "cell phone",
                        "box_index": str(box_index),
                        "x_center": f"{x:.8f}",
                        "y_center": f"{y:.8f}",
                        "width": f"{width:.8f}",
                        "height": f"{height:.8f}",
                        "confidence": f"{score:.6f}",
                    }
                )
        else:
            review_rows.append(
                {
                    "image": path.name,
                    "review_status": status,
                    "class_id": "1",
                    "class_name": "cell phone",
                    "box_index": "",
                    "x_center": "",
                    "y_center": "",
                    "width": "",
                    "height": "",
                    "confidence": "",
                }
            )
        print(f"{status}: {path.name} predicted_boxes={len(boxes)}")

    columns = [
        "image",
        "review_status",
        "class_id",
        "class_name",
        "box_index",
        "x_center",
        "y_center",
        "width",
        "height",
        "confidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"REVIEW OUTPUT: {output_dir}")
    print(f"Preview images and candidate coordinates: {csv_path.name}")
    print("No training labels or dataset files were written.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only dataset audit and unverified phone-box visual review."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="PhoneDetector project directory (defaults to this script's directory).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for candidate previews only (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Inference image size for previews (default: 416).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every valid image/label pair in existing datasets.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    print("PhoneDetector training-input audit")
    print(f"Project root: {root}")
    for relative in (
        Path("Cell-Phone-Dataset/data.yaml"),
        Path("Cell-Phone-Dataset-detect/data.yaml"),
        Path("custom_dataset/roboflow_export/data.yaml"),
        Path("custom_dataset/roboflow_new/data.yaml"),
    ):
        load_and_audit_yaml(root / relative, verbose=args.verbose)

    audit_exact_duplicates(root)

    roboflow_export = root / "custom_dataset" / "roboflow_export"
    export_names = {0: "Object-Detection"}
    print("\n[ROBOFLOW EXPORT TREE: class mapping requires manual confirmation]")
    for split in ("train", "valid", "test"):
        split_root = roboflow_export / split
        audit_pairs(
            f"roboflow_export/{split}",
            split_root / "images",
            split_root / "labels",
            export_names,
            manual_class_review=True,
            verbose=args.verbose,
        )

    roboflow_new = root / "custom_dataset" / "roboflow_new"
    print("\n[ROBOFLOW NEW: kept separate; no labels are imported]")
    print("README.dataset.txt and README.roboflow.txt are metadata and will be ignored.")
    new_yaml_names = load_and_audit_yaml(roboflow_new / "data.yaml")
    audit_pairs(
        "roboflow_new/flat",
        roboflow_new,
        roboflow_new,
        new_yaml_names,
        manual_class_review=True,
        verbose=True,
    )
    for path in sorted(roboflow_new.glob("*.txt")):
        if path.name.casefold() in IGNORED_TEXT_FILES:
            continue
        if is_image_payload(path.read_bytes()):
            print(f"INVALID_IMAGE_DATA_AS_TXT: {path.name}")

    audit_pairs(
        "new_photos/empty_case (negative examples; visual confirmation required)",
        root / "custom_dataset" / "new_photos" / "empty_case",
        root / "custom_dataset" / "new_photos" / "empty_case",
        None,
        manual_class_review=True,
        verbose=True,
    )

    review_phone_predictions(
        root=root,
        output_base=root / "custom_dataset" / "phone_box_review",
        model_path=root / MODEL_RELATIVE_PATH,
        confidence=args.confidence,
        image_size=args.imgsz,
    )

    print("\nAudit finished. No source dataset, main.py, or model was modified.")


if __name__ == "__main__":
    main()