import cv2 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

test_path = Path("./knn_ocr/task")
train_path = test_path / "train"


def extractor(image):
    if image.ndim == 2:
        gray = image
        binary = gray > 0
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 0
    lb = label(binary)
    props = regionprops(lb)
    features = None
    for prop in props:
        if prop.extent > 0.81:
            props.remove(prop)
    
    features = [prop.eccentricity, prop.solidity, prop.extent, prop.perimeter/prop.area,prop.area_convex/prop.area]                                                                                                                               
    return np.array(features,  dtype = "f4")

chararr = []

def make_train(path):
  train = []
  responses = []
  ncls = -1
  for cls in sorted(path.glob("**")):
    ncls += 1
    chararr.append(str(cls)[-1])
    for p in sorted(cls.glob("*.png")):
      train.append(extractor(imread(p)))
      responses.append(ncls)
  train = np.array(train, dtype = "f4").reshape(-1, 5)
  responses = np.array(responses, dtype = 'f4').reshape(-1, 1)
  return train, responses



for i in range(7):
    image = imread(test_path / f"{i}.png")


    train, responses = make_train(train_path)
    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)


    gray = np.mean(image, 2).astype("u1")
    binary = gray > 0
    lb = label(binary.T)
    props = regionprops(lb)

    find = []

    for i, prop in enumerate(props):
        if props[i].extent < 0.7:
            find.append(extractor(props[i].image))
    find = np.array(find, dtype = "f4").reshape(-1,5)

    ret, result, neighbours, dist = knn.findNearest(find,  3)

    for res in result:
        print(chararr[int(res.item())], end="")

    print("\n")
