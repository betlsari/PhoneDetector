from pathlib import Path
from collections import Counter

root = Path("Cell-Phone-Dataset-detect")

for split in ["train", "valid", "test"]:
    label_dir = root / split / "labels"
    counts = Counter()
    errors = []

    for file in label_dir.glob("*.txt"):
        for line_no, line in enumerate(file.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                values = list(map(float, parts[1:]))

                if cls not in (0, 1):
                    errors.append(f"{file.name}:{line_no} sınıf={cls}")
                if len(values) != 4:
                    errors.append(f"{file.name}:{line_no} kutu koordinatı sayısı={len(values)}")
                elif any(v < 0 or v > 1 for v in values):
                    errors.append(f"{file.name}:{line_no} koordinat aralık dışında")

                counts[cls] += 1
            except ValueError:
                errors.append(f"{file.name}:{line_no} okunamadı")

    print(f"\n{split}: sınıf dağılımı {dict(counts)}")
    print("Hata sayısı:", len(errors))
    for error in errors[:5]:
        print(error)
