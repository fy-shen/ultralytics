from ultralytics import YOLO

experiments = [
    {
        "model": "./configs/models/yolo11s.yaml",
        "task": "motion",
        "name": "yolo11s-1280",
        "data": "./configs/dataset/bird_motion.yaml",
        "epochs": 100,
        "imgsz": 1280,
        "batch": 10,
        "weight": "./weights/yolo11s.pt",
    },
]

for exp in experiments:
    model = YOLO(exp["model"], task=exp["task"])
    results = model.train(
        name=exp["name"],
        data=exp["data"],
        epochs=exp["epochs"],
        imgsz=exp["imgsz"],
        batch=exp["batch"],
        model=exp["weight"],
        device=[0],
    )

