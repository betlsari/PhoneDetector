from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from validate_review_export import validate_export
from validate_phone_finetune_dataset import validate_dataset


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Cell-Phone-Dataset-detect"
REVIEW_JSON = Path.home() / "Downloads" / "phone_box_review_approved.json"
PHONE_IMAGES = ROOT / "custom_dataset" / "new_photos" / "phone"
EMPTY_IMAGES = ROOT / "custom_dataset" / "new_photos" / "empty_case"
OUTPUT = ROOT / "custom_dataset" / "phone_finetune_release_02"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = ["0", "cell phone"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_original() -> dict[str, Any]:
    for required in (SOURCE / "data.yaml", REVIEW_JSON, PHONE_IMAGES, EMPTY_IMAGES):
        if not required.exists():
            raise FileNotFoundError(f"Required input is missing: {required}")

    if validate_export(REVIEW_JSON, ROOT) != 0:
        raise ValueError("Reviewed photo export did not pass validation.")

    config = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Original data.yaml must contain a YAML mapping.")
    if config.get("nc") != 2 or config.get("names") != EXPECTED_NAMES:
        raise ValueError(
            f"Unexpected source classes: nc={config.get('nc')!r}, "
            f"names={config.get('names')!r}"
        )

    split_stats: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source {split} image/label directory.")

        images = image_files(image_dir)
        labels = {path.stem.casefold(): path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem.casefold() for path in images}
        missing = [image.name for image in images if image.stem.casefold() not in labels]
        orphaned = sorted(
            label.name for key, label in labels.items() if key not in image_stems
        )
        if missing or orphaned:
            raise ValueError(
                f"{split}: missing labels={missing[:5]}, orphan labels={orphaned[:5]}"
            )

        duplicate_rows = 0
        for image in images:
            digest = sha256(image)
            if digest in source_hashes:
                raise ValueError(
                    f"Exact source image duplicate across splits: {image} and "
                    f"{source_hashes[digest]}"
                )
            source_hashes[digest] = str(image)

            label_path = labels[image.stem.casefold()]
            seen: set[tuple[float, ...]] = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected 5 YOLO fields."
                    )
                try:
                    class_value = float(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"{label_path}:{line_number}: non-numeric YOLO value."
                    ) from exc
                if (
                    not math.isfinite(class_value)
                    or not class_value.is_integer()
                    or int(class_value) not in (0, 1)
                    or not all(math.isfinite(value) for value in values)
                    or not all(0 <= value <= 1 for value in values)
                    or values[2] <= 0
                    or values[3] <= 0
                    or values[0] - values[2] / 2 < 0
                    or values[0] + values[2] / 2 > 1
                    or values[1] - values[3] / 2 < 0
                    or values[1] + values[3] / 2 > 1
                ):
                    raise ValueError(
                        f"{label_path}:{line_number}: invalid class or box."
                    )
                row = tuple(
                    float(np.float32(value)) for value in (class_value, *values)
                )
                if row in seen:
                    duplicate_rows += 1
                seen.add(row)

        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "duplicate_rows_to_remove": duplicate_rows,
        }

    reviewed = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))["reviewed"]
    phone_sources: list[tuple[Path, list[str]]] = []
    for item in reviewed:
        image_path = PHONE_IMAGES / Path(item["image"].replace("\\", "/")).name
        labels = item["yolo_labels"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Reviewed image is missing: {image_path}")
        if item["class_id"] != 1 or item["class_name"] != "cell phone":
            raise ValueError(f"Reviewed image has an incompatible class: {image_path}")
        if not labels or len(labels) != len(item["boxes"]):
            raise ValueError(f"Reviewed image has inconsistent boxes: {image_path}")
        phone_sources.append((image_path, labels))

    empty_sources = image_files(EMPTY_IMAGES)
    if not empty_sources:
        raise ValueError(f"No verified empty-case images found in {EMPTY_IMAGES}")

    return {
        "splits": split_stats,
        "phone_sources": phone_sources,
        "empty_sources": empty_sources,
        "source_hashes": set(source_hashes),
    }


def prepare_dataset() -> Path:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite the existing dataset: {OUTPUT}"
        )

    audited = validate_original()
    OUTPUT.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "source_dataset": str(SOURCE),
        "review_json": str(REVIEW_JSON),
        "splits": audited["splits"],
        "added_reviewed_phone_photos": [],
        "added_verified_empty_case_photos": [],
        "skipped_exact_duplicates": [],
        "removed_exact_duplicate_label_rows": sum(
            split["duplicate_rows_to_remove"]
            for split in audited["splits"].values()
        ),
    }
    known_hashes: set[str] = audited["source_hashes"]

    for split in ("train", "valid", "test"):
        target_images = OUTPUT / "images" / split
        target_labels = OUTPUT / "labels" / split
        target_images.mkdir(parents=True)
        target_labels.mkdir(parents=True)
        source_images = SOURCE / split / "images"
        source_labels = SOURCE / split / "labels"

        for source_image in image_files(source_images):
            target_image = target_images / source_image.name
            target_label = target_labels / f"{source_image.stem}.txt"
            shutil.copy2(source_image, target_image)

            source_label = source_labels / f"{source_image.stem}.txt"
            kept: list[str] = []
            seen: set[tuple[float, ...]] = set()
            for line in source_label.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if not fields:
                    continue
                key = tuple(float(np.float32(float(value))) for value in fields)
                if key not in seen:
                    seen.add(key)
                    kept.append(line.strip())
            target_label.write_text(
                "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
            )

    custom_train_images = OUTPUT / "images" / "train"
    custom_train_labels = OUTPUT / "labels" / "train"

    for role, sources in (
        ("reviewed_phone", audited["phone_sources"]),
        ("verified_empty_case", [(path, []) for path in audited["empty_sources"]]),
    ):
        for source_image, reviewed_labels in sources:
            digest = sha256(source_image)
            if digest in known_hashes:
                manifest["skipped_exact_duplicates"].append(str(source_image))
                continue

            target_image = custom_train_images / source_image.name
            target_label = custom_train_labels / f"{source_image.stem}.txt"
            if target_image.exists() or target_label.exists():
                raise FileExistsError(
                    f"Custom filename conflicts with an existing dataset file: "
                    f"{source_image.name}"
                )
            shutil.copy2(source_image, target_image)

            if role == "reviewed_phone":
                target_label.write_text(
                    "\n".join(reviewed_labels) + "\n", encoding="utf-8"
                )
                manifest["added_reviewed_phone_photos"].append(
                    {
                        "source": str(source_image),
                        "image": target_image.relative_to(OUTPUT).as_posix(),
                        "label": target_label.relative_to(OUTPUT).as_posix(),
                    }
                )
            else:
                target_label.write_text("", encoding="utf-8")
                manifest["added_verified_empty_case_photos"].append(
                    {
                        "source": str(source_image),
                        "image": target_image.relative_to(OUTPUT).as_posix(),
                        "label": target_label.relative_to(OUTPUT).as_posix(),
                    }
                )
            known_hashes.add(digest)

    if manifest["skipped_exact_duplicates"]:
        raise ValueError(
            "An incoming custom photo duplicates an existing image; review its "
            "label/split before building this dataset: "
            f"{manifest['skipped_exact_duplicates']}"
        )

    data_yaml = {
        "path": str(OUTPUT),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": 2,
        "names": EXPECTED_NAMES,
    }
    (OUTPUT / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validate_dataset(OUTPUT)
    print(f"Prepared independent dataset: {OUTPUT}")
    print(f"Original split counts: {manifest['splits']}")
    print(f"Added reviewed phone photos: {len(manifest['added_reviewed_phone_photos'])}")
    print(
        "Added verified empty-case negatives: "
        f"{len(manifest['added_verified_empty_case_photos'])}"
    )
    print(
        "Exact duplicate label rows removed from copied labels: "
        f"{manifest['removed_exact_duplicate_label_rows']}"
    )
    return OUTPUT


if __name__ == "__main__":
    prepare_dataset()
