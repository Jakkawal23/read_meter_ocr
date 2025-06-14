# detect_and_read_digits.py
import cv2
import easyocr
import numpy as np
import re
from ultralytics import YOLO
import uuid

model = YOLO("model/best.pt")
reader = easyocr.Reader(['en'])

def detect_and_read_digits_easy_ocr(file):
    image_data = file.file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model(image)[0]
    if len(results.boxes) == 0:
        return None

    box = results.boxes[0]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    cropped = image[y1:y2, x1:x2]

    # SaveImage
    filename = f"detected/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, cropped)

    ocr_result = reader.readtext(cropped)
    if not ocr_result:
        return None

    text = ocr_result[0][1]

    # 🔧 แปลงข้อความให้เหลือแค่ตัวเลข 0-9
    digits_only = re.sub(r"\D", "", text)

    # 🔢 ตัด/เติมให้เป็น 5 หลัก (ตามความเหมาะสม)
    if len(digits_only) > 5:
        digits_only = digits_only[:5]
    elif len(digits_only) < 5:
        digits_only = digits_only.zfill(5)  # เติม 0 ด้านหน้า

    return digits_only
