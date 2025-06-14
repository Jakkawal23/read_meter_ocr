# ถ้าคุณต้องการใส่ preprocessing เพิ่ม เช่น resize, grayscale
import cv2

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
