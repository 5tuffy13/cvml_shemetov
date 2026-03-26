import cv2
import numpy as np
import matplotlib.pyplot as plt

n = 1000

x1 = 100 + np.random.randint(-25,25,n)
y1 = 100 + np.random.randint(-25,25,n)
r1 = np.repeat(1,n)


x2 = 150 + np.random.randint(-25,25,n)
y2 = 159 + np.random.randint(-25,25,n)
r2 = np.repeat(1,n)

new_point = (127,124)

knn = cv2.ml.KNearest.create()

train = np.stack([np.hstack([x1, x2]), np.hstack([y1,y2])]).T.astype("f4")

responses = np.hstack([r1,r2]).reshape(-1,1).astype("f4")

knn.train(train, cv2.ml.ROW_SAMPLE, responses)

print(train.shape, responses.shape)

ret, results, neighbours, dist = knn.findNearest(np.array(new_point).astype("f4").reshape(1,2), 3)

print(ret,results,neighbours,dist)
plt.scatter(x1,y1,80,"r","^")
plt.scatter(x2,y2,80,"b","s")
marker = "^"
if ret == 2:
    marker = "s"
plt.scatter([new_point[0]], [new_point[1]], 80, "g", marker)
plt.show()