from ultralytics import YOLO


model = YOLO("./configs/models/11/yolo11n.yaml", task="motion")
results = model.train(data="./configs/dataset/bird_motion.yaml", epochs=1, imgsz=640)
