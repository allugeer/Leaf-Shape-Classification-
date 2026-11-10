import json
from pathlib import Path

# 1. 设置路径
root_dir = Path(__file__).parent
class_file = root_dir / "class_names.json"
data_dir = root_dir / "data" / "train"

# 2. 读取类别表
if not class_file.exists():
    print("❌ 找不到 class_names.json 文件！请检查是否在项目根目录。")
    exit()

with open(class_file, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print(f"✅ 成功加载类别表，共 {len(class_names)} 类：{class_names}\n")

# 3. 统计图片数量
total_images = 0
counts = {}

# 支持的图片格式
valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for class_name in class_names:
    class_folder = data_dir / class_name
    
    # 如果文件夹不存在
    if not class_folder.exists():
        print(f"⚠️ 警告：找不到文件夹 {class_folder}")
        counts[class_name] = 0
        continue
    
    # 找出该文件夹下所有合法的图片
    images = [p for p in class_folder.iterdir() if p.suffix.lower() in valid_suffixes]
    counts[class_name] = len(images)
    total_images += len(images)
    
    print(f"类别【{class_name}】：找到 {len(images)} 张图片")

print("-" * 30)
print(f"📊 总计：{total_images} 张图片")