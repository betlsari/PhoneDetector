from pathlib import Path
from PIL import Image, ImageDraw
import math

folder = Path("custom_dataset/images/train")
files = sorted(
    p for p in folder.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
)

w, h, label_h, cols = 240, 200, 28, 3
rows = math.ceil(len(files) / cols)
sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    img = Image.open(path).convert("RGB")
    img.thumbnail((w - 10, h - 10))
    x, y = (i % cols) * w, (i // cols) * (h + label_h)
    sheet.paste(img, (x + (w - img.width) // 2, y + (h - img.height) // 2))
    draw.text((x + 8, y + h), f"{i + 1}. {path.name}", fill="black")

out = Path("custom_dataset/contact_sheet.jpg")
sheet.save(out)
print(f"{len(files)} fotoğraf işlendi: {out.resolve()}")
