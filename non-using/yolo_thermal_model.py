from ultralytics import YOLO

model = YOLO("yolov8s.pt")  

model.train(
    data="Termal/dataset.yaml",
    epochs=80, # Küçük dataset → daha fazla epoch gerekebilir (50-100 civarı).
    imgsz=160, # Verilen görüntülerin upscale olması gerek
    batch=8,
    lr0 = 0.0005,
    lrf = 0.01,
    device="0",
    augment=True,
    cache=True 
)