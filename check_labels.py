from pathlib import Path
from collections import Counter

folder = Path("Cell-Phone-Dataset/train/labels")
counts = Counter()
bad = []

for path in folder.glob("*.txt"):
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue

        try:
            values = [float(x) for x in parts]
        except ValueError:
            bad.append((path.name, line_no, "Sayısal olmayan değer"))
            continue

        if len(values) < 5 or (len(values) - 1) % 2 != 0 and len(values) != 5:
            bad.append((path.name, line_no, f"Beklenmeyen değer sayısı: {len(values)}"))
        elif len(values) == 5:
            counts["Kutu etiketi"] += 1
        else:
            counts["Poligon etiketi"] += 1

print("Etiket özeti:", dict(counts))
print("Hatalı satır sayısı:", len(bad))
for item in bad[:10]:
    print(item)
