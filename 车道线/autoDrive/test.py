from ultralytics import YOLO
# model = YOLO("yolov8n-seg.pt")

model = YOLO(r"C:/Users/Lenovo/Desktop/华清远见/车道线/autoDrive/runs/segment/train3/weights/best.pt")

result = model.predict("./dataset/images/valid/image84.png", save=True,
    imgsz=320,
    conf=0.1,   # 降到 0.01，看是否有微弱响应
    iou=0.45,
    agnostic_nms=False)[0]
result.show()
print(result.boxes)
print("Masks shape:", result.masks.data.shape if result.masks else "None")