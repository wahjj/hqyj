from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

if __name__ == '__main__':
    results = model.train(data="mydata.yaml", epochs=10)