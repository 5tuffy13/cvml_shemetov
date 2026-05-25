import cv2
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# def get_angle(a, b, c):
#     cb = np.atan2(c[1] - b[1], c[0] - b[0])
#     ab = np.atan2(a[1] - b[1], a[0] - b[0])
#     angle = np.rad2deg(cb - ab)
#     angle = angle + 360 if angle < 0 else angle
#     return 360 - angle if angle > 180 else angle

def detect_push_up(keypoints):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    if (left_elbow[1] - left_shoulder[1] > 20 and right_elbow[1] > right_shoulder[1] > 20):
        return False
    
    return True

model = YOLO("yolo26n-pose.pt")
model.to("cpu")

camera = cv2.VideoCapture(0)
ps = None
n_pushups = 0
fl = False

last_seen_time = time.perf_counter()
is_person_visible = False

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break
        
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t_start_inference = time.perf_counter()
    results = model(frame)
    inference_time = time.perf_counter() - t_start_inference

    has_result = len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0

    current_time = time.perf_counter()

    if has_result:
        # есть в кадре
        last_seen_time = current_time
        is_person_visible = True
    else:
        # нет в кадре
        is_person_visible = False
        
        # секундомер
        time_since_last_seen = current_time - last_seen_time
        
        if time_since_last_seen > 5.0: # 5 секунд
            n_pushups = 0
            


    
    annotated = frame.copy()
    if has_result:
        result = results[0]
        keypoints = result.keypoints.xy.tolist()[0]

        annotator = Annotator(annotated)
        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
        

        is_down_phase = detect_push_up(annotated, keypoints, n_pushups)

        if is_down_phase:
            if not fl:
                n_pushups += 1
                fl = True
        else:
            fl = False # прямые руки

        annotated = annotator.result()
        cv2.putText(annotated, f"Push ups : {n_pushups}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 3)

    cv2.imshow("Pose", annotated)

camera.release()
cv2.destroyAllWindows()