from ultralytics import YOLO
import numpy as np
import cv2
import pytesseract
import uuid

model = YOLO("model/water_meter_number/best.pt")

def read_water_meter(file):
    image_data = file.file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model.predict(image, imgsz=640)

    detections = []
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            cls = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            xyxy = box.xyxy.cpu().numpy().tolist()[0]  # [x1, y1, x2, y2]

            x1, y1, x2, y2 = map(int, xyxy)
            digit_crop = image[y1:y2, x1:x2]

            # อ่านเลขจาก Tesseract
            config = "--psm 10 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(digit_crop, config=config).strip()

            # บันทึกรูปเฉพาะตัวเลข
            step_name = f"{i+1}_cls{cls}_conf{int(conf*100)}_digit{text}"
            save_img(step_name, digit_crop)

            # วาดกรอบ + label บนภาพใหญ่
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{text} ({int(conf * 100)}%)"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detections.append({
                "class": cls,
                "confidence": conf,
                "box": xyxy,
                "digit": text
            })

    # เรียงลำดับจากซ้ายไปขวา
    detections = sorted(detections, key=lambda d: d["box"][0])

    # บันทึกภาพที่มีกรอบและข้อความ
    save_img("final", image)

    return {
        "detections": detections,
    }

def save_img(step_name, img):
    filename = f"detected/water_meter/{step_name}_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, img)