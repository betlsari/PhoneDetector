from pathlib import Path

folder = Path("custom_dataset/labels_boxes")
changed = 0

for path in folder.glob("*.txt"):
    lines = path.read_text().splitlines()
    updated = []

    for line in lines:
        parts = line.split()
        if parts:
            parts[0] = "1"
            updated.append(" ".join(parts))

    path.write_text("\n".join(updated) + ("\n" if updated else ""))
    changed += 1

print(f"Tamamlandı: {changed} dosyada sınıf numarası 1 yapıldı.")
