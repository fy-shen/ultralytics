from ultralytics import YOLO


model = YOLO("./configs/models/yolo11s.yaml")
results = model.train(data="./configs/dataset/bird.yaml", epochs=100, imgsz=640, batch=64, device=[0,1])
