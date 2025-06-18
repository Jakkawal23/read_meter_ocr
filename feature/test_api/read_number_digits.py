from ultralytics import YOLO
import numpy as np
import cv2
import uuid

model = YOLO("model/digits/best.pt")

def read_number_digits(file):
    image_data = file.file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    save_img("01_original", image)

    # results = model.predict(image)
    results = model.predict(image, conf=0.25)
    # results = model.predict(image, conf=0.665)
    result_image = image.copy()

    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            # วาดกรอบ
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_id} ({confidence:.2f})"
            cv2.putText(result_image, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "bbox": bbox
            })

    save_img("02_detected", result_image)

    detections = sorted(detections, key=lambda d: d["bbox"][0])

    number = "".join(str(d["class_id"]) for d in detections)

    return {
        "result": detections,
        "number": number
    }

def save_img(step_name, img):
    filename = f"detected/digits/{step_name}_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, img)