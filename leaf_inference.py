"""网页与命令行共用的真实预测，不下载权重或修改模型。"""
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / 'leaf_model.pth'
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def _load_model(path, modified_ns, size):
    # 复用已加载的模型；权重文件改变后重新加载。
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    names = checkpoint['class_names']
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('模型中的类别表无效，请检查模型文件。')
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tuple(names)


def predict_image(image_path, model_path=MODEL_PATH, expected_classes=None):
    """返回类别与各类相对分数；预处理与本次训练/评价保持一致。"""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError('找不到 leaf_model.pth，请将训练好的模型放在项目根目录。')
    info = path.stat()
    model, names = _load_model(str(path), info.st_mtime_ns, info.st_size)
    if expected_classes is not None and set(names) != set(expected_classes):
        raise ValueError('模型类别与界面配置不一致，请核对模型及 class_names.json。')
    with Image.open(image_path) as image:
        inputs = TRANSFORM(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)[0].tolist()
    scores = dict(zip(names, probabilities))
    return {'class_name': max(scores, key=scores.get), 'scores': scores}
