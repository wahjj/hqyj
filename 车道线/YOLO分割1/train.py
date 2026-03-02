from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="coco8-seg.yaml", epochs=30, imgsz=640)