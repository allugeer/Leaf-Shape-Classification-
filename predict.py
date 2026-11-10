import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
from pathlib import Path

# 1. 设置模型路径
MODEL_PATH = "leaf_model.pth"

# 检查模型文件是否存在
if not Path(MODEL_PATH).exists():
    print(f"❌ 找不到模型文件 {MODEL_PATH}，请先运行 train.py 进行训练！")
    exit()

# 2. 加载我们保存的模型和类别顺序
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]

# 3. 重建模型结构（必须和 train.py 保持一致）
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() # 设置为预测模式

# 4. 图片处理规则
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    """预测单张图片的类别"""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ 无法打开图片 {image_path}：{e}")
        return

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    print(f"🍃 图片 [{Path(image_path).name}] 的预测结果是：【{predicted_class}】")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("💡 使用方法: python predict.py <图片路径>")
        print("例如: python predict.py data/train/心形/1.jpg")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)