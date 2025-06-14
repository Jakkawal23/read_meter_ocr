
รับภาพจาก API

ใช้ YOLOv8 ตรวจจับเลขมิเตอร์

ใช้ EasyOCR อ่านเลขจากส่วนที่ตรวจจับ

ส่งค่าที่อ่านได้กลับไปเป็น JSON


--train model
python train_yolo.py


-- run api
uvicorn main:app --reload
