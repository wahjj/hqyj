import numpy as np

from ultralytics import YOLO

classes = {0: "half-ripe", 1: "raw", 2: "ripe"}

# 1、加载模型
# 加载YOLOv8模型
model = YOLO("./runs/detect/train3/weights/best.pt")
model(np.random.rand(100, 100, 3))

def predict_yolo(image):
    # 在帧上运行YOLOv8推理
    results = model(image)
    classes = results[0].boxes.cls
    # 如果有类别名称，可以通过类别索引获取
    class_names = model.names[int(classes)]
    return class_names