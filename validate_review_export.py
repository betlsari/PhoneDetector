from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_REVIEW_COUNT = 18
EXPECTED_CLASS_ID = 1
EXPECTED_CLASS_NAME = "cell phone"
MANUAL_REVIEW_IMAGES = {
    "WhatsApp Image 2026-10-09 at 19.58.07 (2).jpeg": "two-box example",
    "WhatsApp Image 2026-10-09 at 19.58.09.jpeg": "narrow-box example",
}
COORDINATES = ("x_center", "y_center", "width", "height")


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_export(json_path: Path, project_root: Path) -> int:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"ERROR: Cannot read JSON file {json_path}: {exc}")
        return 2
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: Invalid JSON in {json_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        )
        return 2

    if not isinstance(data, dict):
        print("ERROR: JSON root must be an object.")
        return 2

    reviewed = data.get("reviewed")
    source = data.get("source")
    if not isinstance(reviewed, list):
        print("ERROR: JSON field 'reviewed' must be a list.")
        return 2
    if not isinstance(source, str) or not source:
        print("ERROR: JSON field 'source' must be a non-empty path.")
        return 2

    root = project_root.resolve()
    image_root = (root / source).resolve()
    try:
        image_root.relative_to(root)
    except ValueError:
        print(f"ERROR: JSON source path resolves outside the project root: {image_root}")
        return 2

    issues: list[str] = []
    if data.get("format") != "PhoneDetector reviewed annotations":
        issues.append("Unexpected or missing export format.")
    if data.get("version") != 1:
        issues.append("Unexpected or missing export version; expected 1.")
    if len(reviewed) != EXPECTED_REVIEW_COUNT:
        issues.append(
            f"Expected {EXPECTED_REVIEW_COUNT} reviewed images, found {len(reviewed)}."
        )

    seen_images: set[str] = set()
    correct_count = 0
    checked_label_count = 0
    missing_images: list[str] = []
    manual_items: list[tuple[str, str, list[Any]]] = []

    for index, item in enumerate(reviewed, start=1):
        prefix = f"Review {index}"
        if not isinstance(item, dict):
            issues.append(f"{prefix}: entry must be an object.")
            continue

        image = item.get("image")
        if not isinstance(image, str) or not image:
            issues.append(f"{prefix}: 'image' must be a non-empty path.")
            continue

        image_name = Path(image.replace("\\", "/")).name
        if image_name in seen_images:
            issues.append(f"{prefix} ({image_name}): duplicate image entry.")
        seen_images.add(image_name)

        image_path = image_root / image_name
        if not image_path.is_file():
            missing_images.append(image_name)
            issues.append(f"{prefix} ({image_name}): image file is missing at {image_path}.")

        if item.get("decision") == "correct":
            correct_count += 1
        else:
            issues.append(
                f"{prefix} ({image_name}): decision is {item.get('decision')!r}, "
                "expected 'correct'."
            )

        class_id = item.get("class_id")
        if (
            not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or class_id != EXPECTED_CLASS_ID
        ):
            issues.append(
                f"{prefix} ({image_name}): class_id must be {EXPECTED_CLASS_ID}, "
                f"found {class_id!r}."
            )
        if item.get("class_name") != EXPECTED_CLASS_NAME:
            issues.append(
                f"{prefix} ({image_name}): class_name must be "
                f"{EXPECTED_CLASS_NAME!r}, found {item.get('class_name')!r}."
            )

        boxes = item.get("boxes")
        labels = item.get("yolo_labels")
        if not isinstance(boxes, list):
            issues.append(f"{prefix} ({image_name}): 'boxes' must be a list.")
            boxes = []
        if not isinstance(labels, list):
            issues.append(f"{prefix} ({image_name}): 'yolo_labels' must be a list.")
            labels = []
        if len(labels) != len(boxes):
            issues.append(
                f"{prefix} ({image_name}): {len(labels)} YOLO labels for "
                f"{len(boxes)} boxes."
            )

        for box_index, box in enumerate(boxes, start=1):
            box_prefix = f"{prefix} ({image_name}), box {box_index}"
            if not isinstance(box, dict):
                issues.append(f"{box_prefix}: box must be an object.")
                continue
            for coordinate in COORDINATES:
                value = box.get(coordinate)
                if not is_number(value):
                    issues.append(f"{box_prefix}: {coordinate} must be a finite number.")
                elif not 0 <= value <= 1:
                    issues.append(
                        f"{box_prefix}: {coordinate}={value} is outside [0, 1]."
                    )
            for dimension in ("width", "height"):
                value = box.get(dimension)
                if is_number(value) and value <= 0:
                    issues.append(f"{box_prefix}: {dimension} must be positive.")

        for label_index, label in enumerate(labels, start=1):
            checked_label_count += 1
            label_prefix = f"{prefix} ({image_name}), YOLO label {label_index}"
            if not isinstance(label, str):
                issues.append(f"{label_prefix}: label must be a string.")
                continue
            parts = label.split()
            if len(parts) != 5:
                issues.append(
                    f"{label_prefix}: expected 5 fields, found {len(parts)}."
                )
                continue
            try:
                label_class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                issues.append(f"{label_prefix}: class and coordinates must be numeric.")
                continue

            if (
                str(label_class_id) != parts[0]
                or label_class_id != EXPECTED_CLASS_ID
            ):
                issues.append(
                    f"{label_prefix}: class id must be {EXPECTED_CLASS_ID}, "
                    f"found {parts[0]!r}."
                )
            if not all(math.isfinite(value) for value in coordinates):
                issues.append(f"{label_prefix}: coordinates must be finite numbers.")
                continue
            for coordinate_name, value in zip(COORDINATES, coordinates):
                if not 0 <= value <= 1:
                    issues.append(
                        f"{label_prefix}: {coordinate_name}={value} is outside [0, 1]."
                    )
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                issues.append(f"{label_prefix}: width and height must be positive.")

            if label_index <= len(boxes):
                box = boxes[label_index - 1]
                if isinstance(box, dict) and all(
                    is_number(box.get(name)) for name in COORDINATES
                ):
                    if any(
                        abs(float(box[name]) - value) > 0.0000000051
                        for name, value in zip(COORDINATES, coordinates)
                    ):
                        issues.append(
                            f"{label_prefix}: coordinates do not match the corresponding box."
                        )

        if image_name in MANUAL_REVIEW_IMAGES:
            manual_items.append((image_name, MANUAL_REVIEW_IMAGES[image_name], boxes))
            if image_name.endswith("19.58.07 (2).jpeg") and len(boxes) != 2:
                issues.append(
                    f"{image_name}: manual-review example was expected to have "
                    f"2 boxes, found {len(boxes)}."
                )
            if image_name.endswith("19.58.09.jpeg") and len(boxes) != 1:
                issues.append(
                    f"{image_name}: manual-review example was expected to have "
                    f"1 box, found {len(boxes)}."
                )

    print(f"JSON: {json_path.resolve()}")
    print(f"Image source: {image_root}")
    print(f"Reviewed images: {len(reviewed)} (expected {EXPECTED_REVIEW_COUNT})")
    print(f"Existing images: {len(seen_images) - len(missing_images)}/{len(reviewed)}")
    if missing_images:
        print("Missing images:")
        for image_name in missing_images:
            print(f"- {image_name}")
    print(f"Decisions marked correct: {correct_count}/{len(reviewed)}")
    print(f"YOLO labels checked: {checked_label_count}")
    print(f"Validation: {'PASS' if not issues else 'FAIL'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")

    print("Manual review required (reported separately; labels left unchanged):")
    if not manual_items:
        print("- Neither designated example was found.")
    for image_name, reason, boxes in manual_items:
        box_word = "box" if len(boxes) == 1 else "boxes"
        detail = f"{len(boxes)} {box_word}"
        if reason == "narrow-box example" and boxes and isinstance(boxes[0], dict):
            width = boxes[0].get("width")
            detail += f"; box width={width!r}"
        print(f"- {image_name}: {reason} ({detail})")

    print("Training dataset: not modified; training: not started.")
    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a reviewed PhoneDetector JSON export without modifying "
            "annotations or training data."
        )
    )
    parser.add_argument(
        "json_file", type=Path, help="Path to the reviewed annotations JSON export"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root used to locate the JSON source folder",
    )
    args = parser.parse_args()
    return validate_export(args.json_file, args.project_root)


if __name__ == "__main__":
    raise SystemExit(main())
