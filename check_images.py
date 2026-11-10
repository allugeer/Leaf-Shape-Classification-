from pathlib import Path
from PIL import Image

# 指向你放图片的文件夹
sample_dir = Path("data/samples")

if not sample_dir.exists():
    print(f"找不到文件夹：{sample_dir.resolve()}")
    print("请先在项目根目录下创建 data/samples/，把图片放进去。")
    raise SystemExit(1)

# 找出文件夹里所有的图片
image_files = sorted([
    p for p in sample_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
])

if not image_files:
    print(f"在 {sample_dir.resolve()} 里没有找到图片文件。")
    raise SystemExit(1)

ok = 0
bad = []

# 逐个尝试打开图片
for p in image_files:
    try:
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            print(f"OK  {p.name}  尺寸={img.size}  模式={img.mode}")
        ok += 1
    except Exception as e:
        print(f"BAD {p.name}  错误={type(e).__name__}: {e}")
        bad.append((p.name, f"{type(e).__name__}: {e}"))

print("\n===== 汇总 =====")
print(f"图片总数：{len(image_files)}")
print(f"能打开：{ok}")
print(f"打不开：{len(bad)}")