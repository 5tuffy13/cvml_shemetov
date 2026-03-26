import cv2
import numpy as np


def match(gray,template,min_matches=4,thresh=0.5,proj_tresh=7.0):
    sift = cv2.SIFT.create()
    kp1,des1 = sift.detectAndCompute(template,None)
    kp2,des2 = sift.detectAndCompute(gray,None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None,0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)
    if len(good) < 4:
        return None,0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H,mask = cv2.findHomography(src,dst,cv2.RANSAC,proj_tresh)
    if H is None:
        return None,0

    if mask.sum() < min_matches:
        return None,0


    h, w = template.shape[:2]
    corners = np.float32([[0,0],[0,h-1],[w-1, h-1], [w-1, 0]]).reshape(-1,1,2)

    transformed = cv2.perspectiveTransform(corners,H)
    return transformed,mask.sum()
    

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Template", cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 100)

auto_exp = capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
exposure = capture.get(cv2.CAP_PROP_EXPOSURE)
print(f"{auto_exp=}, {exposure=}")


template = None
while True:
  ret, frame = capture.read()
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
    bbox, count = match(gray, template)
    if bbox is not None:
        cv2.polylines(frame, [np.int32(bbox)], isClosed=True, color = (0,250,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("Template", template)
  cv2.imshow("Camera", frame)