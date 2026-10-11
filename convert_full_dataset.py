from pathlib import Path

root = Path("Cell-Phone-Dataset")
out = Path("Cell-Phone-Dataset-detect")

for split in ("train", "valid", "test"):
    src_images = root / split / "images"
    src_labels = root / split / "labels"
    dst_images = out / split / "images"
    dst_labels = out / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if src_images.exists():
        import shutil
        for image in src_images.iterdir():
            if image.is_file():
                shutil.copy2(image, dst_images / image.name)

    if not src_labels.exists():
        continue

    for path in src_labels.glob("*.txt"):
        converted = []

        for line in path.read_text().splitlines():
            p = line.split()
            if not p:
                continue

            cls = p[0]
            v = list(map(float, p[1:]))

            if len(v) == 4:
                x, y, w, h = v
            elif len(v) >= 6 and len(v) % 2 == 0:
                xs, ys = v[0::2], v[1::2]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                x, y = (xmin+xmax)/2, (ymin+ymax)/2
                w, h = xmax-xmin, ymax-ymin
            else:
                raise ValueError(f"Beklenmeyen etiket: {path.name}: {line}")

            converted.append(f"{cls} {x} {y} {w} {h}")

        (dst_labels / path.name).write_text(
            "\n".join(converted) + ("\n" if converted else "")
        )

print("Dönüştürme tamamlandı:", out.resolve())
