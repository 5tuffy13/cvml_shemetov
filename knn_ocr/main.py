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
    props = regionprops(lb.T)
    features = None
    for prop in props:

        features = [prop.eccentricity, prop.extent, prop.area_convex/prop.area, prop.axis_minor_length/prop.perimeter * 3, prop.perimeter * 10**-4 * 5]                                                                                                                               
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
    print(f"{i}. ", end="")
    image = imread(test_path / f"{i}.png")


    train, responses = make_train(train_path)
    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)


    gray = np.mean(image, 2).astype("u1")
    binary = gray > 0
    lb = label(binary.T)
    props = regionprops(lb)

    toDelete = []
    for prop in props:
        if prop.extent > 0.7 and prop.axis_minor_length/prop.perimeter > 0.2:
            toDelete.append(prop)

    for item in toDelete:
        props.remove(item)

    find = []
    spacearr = []
    for j, prop in enumerate(props):
        if prop != props[-1]:
            if props[j+1].bbox[0] > props[j].bbox[2] + 20:
                spacearr.append(j+1)
        find.append(extractor(props[j].image))

    find = np.array(find, dtype = "f4").reshape(-1,5)

    ret, result, neighbours, dist = knn.findNearest(find,  3)

    k = 0
    for res in result:
        
        if k in spacearr:
            print(" ", end="")
        print(chararr[int(res.item())], end="")
        k+=1

    print("\n")
