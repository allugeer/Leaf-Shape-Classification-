import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import csv
from pathlib import Path

# 配置
MODEL_PATH = "leaf_model.pth"
# 这里我们暂时评估 data/train 文件夹里的所有图片
# 等以后 A 给了专门的“测试集”文件夹，把这里改成 "data/test" 即可
TEST_DIR = Path("data/train") 
OUTPUT_CSV = "result.csv"

# 1. 加载模型
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. 图片处理规则
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. 批量测试
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
records = []

print("⏳ 正在批量预测中，请稍候...")

for class_name in class_names:
    folder = TEST_DIR / class_name
    if not folder.exists():
        continue
    
    # 遍历该类别下的所有图片
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue
            
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
        predicted_class = class_names[predicted_idx.item()]
        is_correct = "是" if predicted_class == class_name else "否"
        
        # 记录：图片名字，真实类别，AI预测类别，是否预测正确
        records.append([img_path.name, class_name, predicted_class, is_correct])

# 4. 统计准确率并写入 CSV (Excel 可打开)
if len(records) > 0:
    # 写入表格
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["图片名字", "真实类别", "AI预测类别", "是否预测正确"])
        writer.writerows(records)
        
    correct_count = sum(1 for r in records if r[3] == "是")
    acc = (correct_count / len(records)) * 100
    
    print("\n✅ 批量测试完成！")
    print(f"📊 总计测试图片: {len(records)} 张")
    print(f"🎯 预测正确: {correct_count} 张")
    print(f"🏆 整体准确率: {acc:.2f}%")
    print(f"💾 详细的逐图预测结果已保存到根目录的【{OUTPUT_CSV}】")
else:
    print("❌ 没有找到任何图片，请检查 data/train 文件夹。")