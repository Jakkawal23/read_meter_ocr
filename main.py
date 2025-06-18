# main.py
from fastapi import FastAPI, UploadFile, File

from feature.test_api.read_electric_meter import read_electric_meter
from feature.test_api.read_water_meter import read_water_meter
from feature.test_api.read_qr_code import read_qr_code
from feature.test_api.read_number_digits import read_number_digits

app = FastAPI()

#Detect electric meter
@app.post("/read-electric-meter")
async def detect_electric_meter(file: UploadFile = File(...)):
    result = read_electric_meter(file)
    return result



#Detect electric meter
@app.post("/test/read-electric-meter")
async def detect_electric_meter(file: UploadFile = File(...)):
    result = read_electric_meter(file)
    return result

#Detect water meter
@app.post("/test/read-water-meter")
async def detect_water_meter(file: UploadFile = File(...)):
    result = read_water_meter(file)
    return result

#Detect qr code meter
@app.post("/test/read-qr-code")
async def detect_qr_code(file: UploadFile = File(...)):
    result = read_qr_code(file)
    return result

#Detect number digits
@app.post("/test/read-number-digits")
async def detect_number_digits(file: UploadFile = File(...)):
    result = read_number_digits(file)
    return result