"""使用与网页一致的模型和图片处理进行单图预测。"""
import sys
from leaf_inference import predict_image

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('使用方法: python predict.py "图片路径"')
        raise SystemExit(1)
    try:
        result = predict_image(sys.argv[1])
        print(f"预测类别：{result['class_name']}")
        for name, score in result["scores"].items():
            print(f"{name}：{score:.1%}")
    except Exception as error:
        print(f"预测失败：{error}", file=sys.stderr)
        raise SystemExit(1)
