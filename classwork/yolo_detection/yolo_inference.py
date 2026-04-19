from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib import patches
import numpy as np

classes = {0: "cube", 1:"sphere"}

image_path = "/Users/alexander/study/cvml_shemetov/classwork/yolo_detection/dataset/images/8cde5b34-photo_11_2026-03-29_12-35-23.jpg"
model = YOLO("/Users/alexander/study/cvml_shemetov/runs/detect/figures/yolo/weights/best.pt")

plt.subplot(111)
image = np.array(Image.open(image_path).convert("RGB"))
plt.imshow(image)

result = model.predict(source = image_path, conf = 0.1, iou = 0.1, imgsz = 640)[0]
boxes = result.boxes.xyxy.cpu().numpy()
cls = result.boxes.cls.cpu().numpy()
scores = result.boxes.conf.cpu().numpy()

for box, label, score in zip(boxes, cls, scores):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1,y1), x2-x1, y2-y1, linewidth = 2
    )
    plt.gca().add_patch(rect)
    plt.gca().text(x1, y1-10, f"{score:.2f}", color = "white", fontsize = 12)
plt.show()

