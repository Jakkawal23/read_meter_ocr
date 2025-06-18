# main.py
from fastapi import FastAPI, UploadFile, File

from read_electric_meter import read_electric_meter
from read_water_meter import read_water_meter
from read_qr_code import read_qr_code
from read_number_digits import read_number_digits

app = FastAPI()

@app.post("/read-electric-meter")
async def detect_electric_meter(file: UploadFile = File(...)):
    result = read_electric_meter(file)
    return result

@app.post("/read-water-meter")
async def detect_water_meter(file: UploadFile = File(...)):
    result = read_water_meter(file)
    return result

@app.post("/read-qr-code")
async def detect_qr_code(file: UploadFile = File(...)):
    result = read_qr_code(file)
    return result

@app.post("/read-number-digits")
async def detect_number_digits(file: UploadFile = File(...)):
    result = read_number_digits(file)
    return result