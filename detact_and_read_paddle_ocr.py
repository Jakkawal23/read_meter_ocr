import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR

# โหลด YOLOv8 model
model = YOLO("model/best.pt")

# โหลด PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')  # หรือ 'en' ก็เพียงพอถ้ามีแต่เลข

def detect_and_read_digits_paddleocr(file):
    # แปลงรูปจาก file เป็น image
    image_data = file.file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # ตรวจจับกรอบเลขด้วย YOLOv8
    results = model(image)[0]
    if len(results.boxes) == 0:
        return None

    # ใช้ box แรกสุดที่ detect ได้
    box = results.boxes[0]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    cropped = image[y1:y2, x1:x2]

    # OCR ด้วย PaddleOCR
    ocr_result = ocr.ocr(cropped, cls=False)

    if not ocr_result or not ocr_result[0]:
        return None

    # ดึงข้อความที่อ่านได้
    text = ocr_result[0][0][1][0]

    # 🔧 กรองให้เหลือเฉพาะตัวเลข
    digits_only = re.sub(r"\D", "", text)

    # 🔢 ตัด/เติมให้เป็น 5 หลัก
    if len(digits_only) > 5:
        digits_only = digits_only[:5]
    elif len(digits_only) < 5:
        digits_only = digits_only.zfill(5)

    return digits_only
