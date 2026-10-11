from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = ["0", "cell phone"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_path(root: Path, config: dict[str, Any], value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        dataset_root = Path(str(config.get("path", root)))
        if not dataset_root.is_absolute():
            dataset_root = root / dataset_root
        path = dataset_root / path
    return path.resolve()


def parse_label(
    path: Path, *, custom_phone: bool = False
) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected exactly 5 fields.")
        try:
            class_value = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric field.") from exc
        if not math.isfinite(class_value) or not class_value.is_integer():
            raise ValueError(f"{path}:{line_number}: class id must be an integer.")
        class_id = int(class_value)
        if class_id not in (0, 1) or (custom_phone and class_id != 1):
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}.")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"{path}:{line_number}: coordinates must be finite in [0,1].")
        x_center, y_center, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: box dimensions must be positive.")
        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            raise ValueError(f"{path}:{line_number}: box exceeds image bounds.")
        rows.append((class_id, *values))
    return rows


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    root = dataset_dir.resolve()
    yaml_path = root / "data.yaml"
    manifest_path = root / "source_manifest.json"
    if not yaml_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing data.yaml or manifest in {root}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("data.yaml must contain a mapping.")
    if config.get("nc") != 2 or config.get("names") != EXPECTED_NAMES:
        raise ValueError(
            f"Class mapping mismatch: nc={config.get('nc')!r}, "
            f"names={config.get('names')!r}"
        )

    all_hashes: dict[str, str] = {}
    stats: dict[str, Any] = {}
    split_images: dict[str, set[str]] = {}
    for split, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_dir = split_path(root, config, config[yaml_key])
        label_dir = image_dir.parent.parent / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} images/labels directory.")
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = {path.stem.casefold(): path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem.casefold() for path in images}
        if len(image_stems) != len(images):
            raise ValueError(f"{split}: duplicate image stems.")
        missing = [image.name for image in images if image.stem.casefold() not in labels]
        orphaned = [path.name for key, path in labels.items() if key not in image_stems]
        if missing or orphaned:
            raise ValueError(
                f"{split}: missing labels={missing[:5]}, orphan labels={orphaned[:5]}"
            )

        empty_labels = 0
        objects = {0: 0, 1: 0}
        for image in images:
            label = labels[image.stem.casefold()]
            try:
                with Image.open(image) as picture:
                    picture.verify()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable image: {image}") from exc
            digest = sha256(image)
            if digest in all_hashes:
                raise ValueError(
                    f"Duplicate image across dataset splits: {image} and "
                    f"{all_hashes[digest]}"
                )
            all_hashes[digest] = str(image)

            rows = parse_label(label)
            if not rows:
                empty_labels += 1
            numeric_rows = [
                tuple(float(np.float32(value)) for value in row) for row in rows
            ]
            if len(set(numeric_rows)) != len(rows):
                raise ValueError(f"{label}: duplicate YOLO label rows remain.")
            for row in rows:
                objects[row[0]] += 1

        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_labels,
            "objects": objects,
        }
        split_images[split] = image_stems

    custom_roles = (
        ("added_reviewed_phone_photos", "phone"),
        ("added_verified_empty_case_photos", "empty"),
    )
    custom_counts: dict[str, int] = {}
    for manifest_key, role in custom_roles:
        records = manifest.get(manifest_key)
        if not isinstance(records, list):
            raise ValueError(f"Manifest field {manifest_key!r} must be a list.")
        custom_counts[role] = len(records)
        for record in records:
            image = (root / record["image"]).resolve()
            label = (root / record["label"]).resolve()
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"Manifest image/label missing: {record}")
            if image.parent != split_path(root, config, config["train"]):
                raise ValueError(f"Custom image is not in the training split: {image}")
            if role == "phone":
                rows = parse_label(label, custom_phone=True)
                if not rows:
                    raise ValueError(f"Reviewed phone photo has an empty label: {image}")
            elif label.read_text(encoding="utf-8").strip():
                raise ValueError(f"Verified empty-case label must be empty: {label}")

    if custom_counts != {"phone": 18, "empty": 3}:
        raise ValueError(f"Unexpected custom photo counts: {custom_counts}")

    result = {
        "dataset": str(root),
        "class_mapping": config["names"],
        "splits": stats,
        "custom_photos": custom_counts,
        "duplicate_label_rows_removed": manifest.get(
            "removed_exact_duplicate_label_rows"
        ),
        "status": "PASS",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate image-label pairs in the independent phone dataset."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "custom_dataset"
        / "phone_finetune_release_02",
    )
    validate_dataset(parser.parse_args().dataset)
