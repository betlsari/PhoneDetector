from pathlib import Path

source = Path("custom_dataset/roboflow_export/train/labels")
target = Path("custom_dataset/labels_boxes")
target.mkdir(parents=True, exist_ok=True)

converted = 0

for path in source.glob("*.txt"):
    output = []

    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue

        cls = values[0]
        coords = list(map(float, values[1:]))

        if len(coords) == 4:
            x, y, w, h = coords
        else:
            xs = coords[0::2]
            ys = coords[1::2]

            if len(xs) < 3 or len(xs) != len(ys):
                continue

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

        output.append(f"{cls} {x} {y} {w} {h}")

    (target / path.name).write_text(
        "\n".join(output) + ("\n" if output else "")
    )
    converted += 1

print(f"Tamamlandı: {converted} etiket dosyası dönüştürüldü.")
print(f"Yeni etiket klasörü: {target.resolve()}")
