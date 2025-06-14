# main.py
from fastapi import FastAPI, UploadFile, File
from detect_and_read_tessaract import detect_and_read_digits_tessaract
from detect_and_read_easy_ocr import detect_and_read_digits_easy_ocr
# from detect_and_read_tessaract_ocr import detect_and_read_digits_paddleocr

app = FastAPI()

@app.post("/read-meter/")
async def read_meter(file: UploadFile = File(...)):
    # value = detect_and_read_digits_paddleocr(file)
    # value = detect_and_read_digits_easy_ocr(file)
    value = detect_and_read_digits_tessaract(file)
    return value
