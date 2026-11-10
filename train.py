import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path
import time

# ---------------- 配置参数 ----------------
DATA_DIR = Path("data/train")
MODEL_SAVE_PATH = "leaf_model.pth"
BATCH_SIZE = 4
EPOCHS = 30          # 从 5 改成了 30，让 AI 多学几遍
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 正在使用的计算设备: {device}")

# ---------------- 1. 准备数据 (加入数据增强) ----------------
# 训练用的处理：随机裁剪、随机翻转、颜色微调（让AI多见见世面）
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# 验证用的处理：只缩放（保持原样）
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

full_dataset = datasets.ImageFolder(root=DATA_DIR)
# 划分数据集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 给训练集和验证集分别加上不同的处理规则
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"📊 数据划分完毕：训练集 {train_size} 张，验证集 {val_size} 张")

# ---------------- 2. 建立模型 ----------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ---------------- 3. 开始训练 ----------------
print("\n🚀 开始训练...")
start_time = time.time()
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total if total > 0 else 0
    
    # 如果这次比之前好，就保存一下
    if val_acc >= best_acc:
        best_acc = val_acc
        checkpoint = {"model_state_dict": model.state_dict(), "class_names": class_names}
        torch.save(checkpoint, MODEL_SAVE_PATH)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | 训练误差: {running_loss/len(train_loader):.4f} | 验证准确率: {val_acc:.2f}% | 历史最佳: {best_acc:.2f}%")

end_time = time.time()
print(f"✅ 训练完成！耗时: {end_time - start_time:.2f} 秒")
print(f"💾 表现最好的模型已保存到 {MODEL_SAVE_PATH}")