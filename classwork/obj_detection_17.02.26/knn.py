import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

def extractor(image):
  if image.ndim == 2:
    binary = image
  else:
    gray = np.mean(image, 2).astype("u1")
    threshold = 255
    binary = gray < threshold
  lb = label(binary)
  props = regionprops(lb)
  plt.imshow(lb)
  return np.array([props[0].eccentricity, (props[0].area / np.pi) ** 0.5], dtype = 'f4')

def make_train(path):
  train = []
  responses = []
  ncls = 0
  for cls in sorted(path.glob("*")):
#     print(cls)
    ncls += 1
    for p in cls.glob("*.png"):
#       print(p)
      train.append(extractor(imread(p)))
      responses.append(ncls)
  train = np.array(train, dtype = "f4").reshape(-1, 2)
  responses = np.array(responses, dtype = 'f4').reshape(-1, 1)
  return train, responses

data = Path("./obj_detection_17.02.26/out/")
image = imread(data / 'image.png')

train, responses = make_train(data / "train")
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

gray = image.mean(2)
binary = gray < 255
lb = label(binary)
props = regionprops(lb)

find = []

for i, prop in enumerate(props):
  find.append(extractor(props[i].image))
find = np.array(find, dtype = "f4").reshape(-1,2)


ret, result, neighbours, dist = knn.findNearest(find,  5)
print(ret, result, neighbours, dist)
# print(train, response)
# print(extractor(image)) 
plt.imshow(image)
plt.show()
