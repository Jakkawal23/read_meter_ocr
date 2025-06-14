from ultralytics import YOLO

# ใช้โมเดล YOLO ที่เล็กสุด (หรือใช้ yolov8s.pt, yolov8m.pt ตามต้องการ)
model = YOLO("yolov8n.pt")

# เทรนโมเดล
model.train(
    data="dataset/data.yaml",        # path ไปยัง data.yaml ที่อธิบาย dataset
    epochs=100,
    imgsz=640,
    project="runs/train",
    name="meter-model"
)
