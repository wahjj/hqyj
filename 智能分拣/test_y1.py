from ultralytics import YOLO
import os

# 路径配置
model_path = r"runs/detect/train3/weights/best.pt"
img_path = "dataset_y1/test/images/img20.jpg"

# 检查文件是否存在
assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
assert os.path.exists(img_path), f"图像文件不存在: {img_path}"

# 加载模型并预测
model = YOLO(model_path)
results = model.predict(
    source=img_path,
    save=True,
    conf=0.25,
    imgsz=640
)

print("✅ 预测完成！结果已保存到 'runs/detect/predict/' 目录")