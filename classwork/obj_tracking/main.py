import numpy as np
import cv2
from pathlib import Path
from collections import deque

crossing = deque(maxlen=40)

def is_crossed(y_current, direction, frame_id):
    for f,y,d in crossing:
        if d == direction and abs(y-y_current) < 40:
            if frame_id - f <= crossing.maxlen:
                return True
    return False
image_path = Path("./CVKiselev/obj_tracking/pedestrians")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

corner_params = {"maxCorners": 100,
                 "qualityLevel":0.1,
                 "minDistance": 15,
                 "blockSize": 15,
                 }


op_params = {"winSize": (100,100),
             "maxLevel": 2,
             "criteria": (cv2.TermCriteria_COUNT |
                          cv2.TERM_CRITERIA_EPS
                          ,10, 0.03)
             }

count_left_to_right = 0
count_right_to_left = 0

prev_gray = None

LINE_X = 240
pts = None
good_new = []
good_old = []
for frame_id, path in enumerate(sorted(image_path.glob("*.jpg"))):
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, **corner_params)
    if pts is None or len(pts) < 20:
        pts = cv2.goodFeaturesToTrack(gray, **corner_params)

    if prev_gray is not None:
        op_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **op_params)

        for p in op_pts:
            cv2.circle(frame, tuple(p.flatten().astype(int)), 5, (0,255,0), -1)

        good_new, good_old = [],[]

        for i, (new_pt, old_pt) in enumerate(zip(op_pts, pts)):
            if status[i] == 1 and error[i] < 5.0:
                new_pt = new_pt.flatten()
                old_pt = old_pt.flatten()
                dx = new_pt[0] - old_pt[0]
                dy = new_pt[1] - old_pt[1]
                movement = np.hypot(dx,dy)
                if movement > 0.1:
                    good_new.append(new_pt)
                    good_old.append(old_pt)


        good_new = np.array(good_new)
        good_old = np.array(good_old)


        for (x_new, y_new), (x_old, y_old) in zip(good_new, good_old):
            dx = x_new - x_old
            if x_old < (LINE_X+5) and (LINE_X-5) <= x_new and dx > 0.3:
                if not is_crossed(y_new, 1, frame_id):
                    print ("L to r")
                    count_left_to_right+=1
                    crossing.append((frame_id,y_new, 1))
            if x_old > (LINE_X-5) and (LINE_X+5) >= x_new and dx < -0.3:
                if not is_crossed(y_new, -1, frame_id):
                    print("R to L")
                    count_right_to_left+=1
                    crossing.append((frame_id, y_new, -1))
    

        for p in good_new:
            cv2.circle(frame, tuple(p.flatten().astype(int)), 3, (255,0,0), -1)




    for p in pts:
        cv2.circle(frame, tuple(p.flatten().astype(int)), 1, (0,0,255), -1)


    prev_gray = gray.copy()
    pts = good_new.reshape(-1,1,2) if len(good_new) > 0 else None
    cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (255,0,0),2)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(int(1000/120))
    if (key & 0xFF) == 27:
        break

print("L to R:", count_left_to_right)
print("R to L:", count_right_to_left)
