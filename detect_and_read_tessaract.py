# detect_and_read_digits.py
import cv2
import numpy as np
import re
import pytesseract
from ultralytics import YOLO
import uuid
import json

import easyocr

model = YOLO("model/best.pt")

reader = easyocr.Reader(['en'])

custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'

def detect_and_read_digits_tessaract(file):
    image_data = file.file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Read QR Code
    qr_decoder = cv2.QRCodeDetector()
    qr_data, _, _ = qr_decoder.detectAndDecode(image)
    try:
        qr_obj = json.loads(qr_data) if qr_data else {}
    except json.JSONDecodeError:
        qr_obj = {"raw": qr_data.strip()} if qr_data else {}

    results = model(image)[0]
    if len(results.boxes) == 0:
        return {
            "tesseract": {
                "raw": "",
                "digits": ""
            },
            "easyocr": {
                "raw": "",
                "digits": ""
            },
            "qr_code": qr_obj
        }

    # Crop detected area (only the first box)
    box = results.boxes[0]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    cropped = image[y1:y2, x1:x2]

    # Preprocess image
    fianl = adjust_image(cropped)

    # OCR by Tesseract
    text_01 = pytesseract.image_to_string(fianl, config=custom_config)
    digits_01 = ""
    if text_01:
        digits_01 = re.sub(r"\D", "", text_01)

    # OCR by EasyOCR
    ocr_result = reader.readtext(fianl)
    if ocr_result:
        text_02 = ocr_result[0][1].strip()
        digits_02 = re.sub(r"\D", "", text_02)
    else:
        text_02 = ""
        digits_02 = ""

    # if len(digits_only) >= 5:
    #     digits_only = digits_only[-5:]
    # else:
    #     digits_only = digits_only.zfill(5)

    return {
        "tesseract": {
            "raw": text_01,
            "digits": digits_01
        },
        "easyocr": {
            "raw": text_02,
            "digits": digits_02
        },
        "qr_code": qr_obj
    }



def save_img(step_name, img):
    filename = f"detected/{step_name}_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, img)

def adjust_image(cropped):
    # Step 1: Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    save_img("01_gray", gray)

    # Step 2: CLAHE (เพิ่ม contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    save_img("02_enhanced", enhanced)

    # Step 3: Threshold (ไม่ invert ตอนนี้)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_img("03_thresh", thresh)

    # Step 4: Morphology - เปิดเพื่อกำจัดจุดเล็กๆ (noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    save_img("04_opened", opened)

    # Step 5: หา contours แล้วเก็บเฉพาะตัวเลขใหญ่
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opened)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 150 and h > 15 and w > 8:  # ปรับให้ยืดหยุ่นขึ้น
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    save_img("05_mask_filtered", mask)

    # Step 6: Apply mask ถ้าไม่ว่าง
    if np.count_nonzero(mask) == 0:
        print("⚠️ ไม่มี contour ผ่านเงื่อนไข — ใช้ opened แทน")
        filtered = opened
    else:
        filtered = cv2.bitwise_and(opened, mask)
    save_img("06_filtered", filtered)

    # Step 7: Invert เพื่อให้พื้นขาว ตัวเลขดำ (เหมาะกับ OCR)
    inverted = cv2.bitwise_not(filtered)
    save_img("07_inverted", inverted)

    # Step 8: Resize
    resized = cv2.resize(inverted, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    save_img("08_resized", resized)

    # Step 9: Sharpen
    sharpen_kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
    save_img("09_sharpened", sharpened)

    return sharpened