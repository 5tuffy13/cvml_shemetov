from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib import patches
import numpy as np
import cv2
classes = {0: "cube", 1:"neither", 2:"sphere"}
image_path = "/Users/alexander/study/cvml_shemetov/classwork/yolo_detection/dataset/images/8cde5b34-photo_11_2026-03-29_12-35-23.jpg"
model = YOLO("/Users/alexander/study/cvml_shemetov/runs/detect/figures/yolo8/weights/best.pt")


cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

cls = None
while True:
    _,frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == 27:
        break
    
    result = model.predict(source = image, conf = 0.1, iou = 0.1, imgsz = 640)[0]
    if len(result.boxes.xyxy.cpu().numpy()) > 0:
        box = result.boxes.xyxy.cpu().numpy()[0] 
        cls = result.boxes.cls.cpu().numpy()[0]
        scores = result.boxes.conf.cpu().numpy()[0]
    
        x1, y1, x2, y2 = box.astype(int)
    if cls is not None:
        cv2.putText(frame, f"{model.names[cls]}", (x1+20,y1), 3, 3, (0,255,0))
        cv2.putText(frame, f"{scores:.2f}", (x1+20,y1-60), 3, 3, (0,255,0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,230,0))
    cv2.imshow("class", frame)
    
    cls = None
