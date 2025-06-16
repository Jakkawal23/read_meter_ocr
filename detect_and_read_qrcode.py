# detect_and_read_qrcode.py
import cv2
import numpy as np
from ultralytics import YOLO
import uuid

# โหลด YOLO model สำหรับตรวจหา QR Code
model_qr = YOLO("model/qr_code/best.pt")

def detect_and_read_qrcode(image):
    # use to test api
    # image_data = file.file.read()
    # image_np = np.frombuffer(image_data, np.uint8)
    # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model_qr(image)[0]

    if len(results.boxes) == 0:
        return {
            "qr_code": "",
            "qr_image": None
        }

    # ตัดภาพจาก bounding box อันแรก
    box = results.boxes[0]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    cropped_qr = image[y1:y2, x1:x2]
    save_img("qr_cropped_01", cropped_qr)

    processed_qr = preprocess_qr_image(cropped_qr, scale=3)

    qr_decoder = cv2.QRCodeDetector()
    qr_data, points, _ = qr_decoder.detectAndDecode(processed_qr)

    return {
        "qr_code": qr_data
    }




def preprocess_qr_image(image, scale=3):
    resized = resize_qr_image(image, scale=scale)
    binary = threshold_otsu(resized)
    return binary

def resize_qr_image(image, scale=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ขยายภาพแบบไม่เบลอ (pixel perfect)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    save_img("qr_cropped_03_gray", gray)
    save_img("qr_cropped_04_resized", resized)

    return resized

def threshold_otsu(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_img("qr_cropped_05_binary", binary)
    return binary


def save_img(step_name, img):
    filename = f"detected/qr_code/{step_name}_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, img)