from ultralytics import YOLO

model = YOLO(r"C:/Users/Lenovo/Desktop/华清远见/车道线/autoDrive/runs/segment/train3/weights/best.pt")

results = model.val(data="mydata.yaml")