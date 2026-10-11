from pathlib import Path

content = """train: ../Cell-Phone-Dataset-detect/train/images
val: ../Cell-Phone-Dataset-detect/valid/images
test: ../Cell-Phone-Dataset-detect/test/images

nc: 2
names: ['0', 'cell phone']
"""

path = Path("Cell-Phone-Dataset-detect/data.yaml")
path.write_text(content, encoding="utf-8")
print("Oluşturuldu:", path.resolve())
print(path.read_text(encoding="utf-8"))
