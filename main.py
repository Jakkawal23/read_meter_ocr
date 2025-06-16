# main.py
from fastapi import FastAPI, UploadFile, File
from detect_and_read_meter import detect_and_read_meter

from detect_and_read_qrcode import detect_and_read_qrcode

app = FastAPI()

@app.post("/read-meter/")
async def read_meter(file: UploadFile = File(...)):
    # value = detect_and_read_digits_paddleocr(file)
    # value = detect_and_read_digits_easy_ocr(file)
    value = detect_and_read_meter(file)
    return value

@app.post("/read-qr-code/")
async def read_meter(file: UploadFile = File(...)):
    qr_result = detect_and_read_qrcode(file)
    return qr_result

