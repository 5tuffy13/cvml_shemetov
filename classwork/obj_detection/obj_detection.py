# Template matching

import cv2
import numpy as np


def overlap(box1, box2, threshold):  
    x1_min, y1_min = box1["top_left"]
    x1_max, y1_max = box1["bottom_right"]
    x2_min, y2_min = box2["top_left"]
    x2_max, y2_max = box2["bottom_right"]


    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if (inter_x_max < inter_x_min or inter_y_max < inter_y_min):
        return False
    

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area) # intersection over union

    return iou > threshold


def non_max_supression(boxes, overlap_threshold = 0.3):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key = lambda item : item["confidence"], reverse = True)

    picked = []
    while boxes:
        current = boxes.pop(0)
        picked.append(current)
        boxes = [box for box in boxes if not overlap(current, box, overlap_threshold)]

    return picked

def match(image, template, scales = np.arange(0.4, 1.7, 0.1), threshold = 0.9):
    matches = []
    for scale in scales:
        resized_template = cv2.resize(template, (int(template.shape[0] * scale), int(template.shape[1] * scale)))
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        location = np.where(result >= threshold)
        for point in zip(*location[::-1]):
            matches.append({
                "top_left" : point,
                "bottom_right" : ((point[0] + int(template.shape[1] * scale)), point[1] + int(template.shape[0] * scale)),
                "confidence" : result[point[1], point[0]],
                "scale" : scale
            })
    return non_max_supression(matches)


cv2.namedWindow("Camera")
cv2.namedWindow("Template")

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
capture.set(cv2.CAP_PROP_EXPOSURE, -5)
    
auto_exp = capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)

exposure = capture.get(cv2.CAP_PROP_EXPOSURE)

print(f"{auto_exp =}, {exposure=}")



template = None
while True:
    ret, frame = capture.read(0)
    frame = cv2.resize(frame, (1600, 900), fx = 0.5, fy = 0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break
    elif chr(key) == "t":
        roi = cv2.selectROI("ROI", gray)
        template = gray[roi[1]:roi[1]+roi[3],
                    roi[0]:roi[0]+roi[2]]
        cv2.destroyWindow("ROI")

    if template is not None:
    # cv2.imshow("Template", template)
        matches = match(gray, template, threshold = 0.83)
        for m in matches:
            cv2.rectangle(frame, m["top_left"], m["bottom_right"], (0, 255, 0), 2)
            cv2.putText(frame, f"{m['confidence']:.2f}", m["top_left"], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    cv2.imshow("Camera", frame)

    


# image = cv2.imread("./circles.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# x, y, width, height = cv2.selectROI("Template", gray)
# template = gray[y:y+height, x:x+width]



# cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)


# matches = match(gray, template)
# print(matches)

# for match in matches:
#     cv2.rectangle(image, match["top_left"], match["bottom_right"], (0,255, 0),1)
#     cv2.putText(image, f"{match["confidence"]:.2f}", match["top_left"],cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)

# print(len(matches))

# cv2.imshow("Image", image)
# cv2.waitKey()