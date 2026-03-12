import os
import cv2
import numpy as np

FRAME_PATH = "frames"

data = []
labels = []

for img in os.listdir(FRAME_PATH):

    path = os.path.join(FRAME_PATH, img)

    image = cv2.imread(path)

    image = cv2.resize(image, (224,224))

    image = image / 255.0

    data.append(image)

    if "fake" in img:
        labels.append(1)
    else:
        labels.append(0)

X = np.array(data)
y = np.array(labels)

print("Dataset Loaded")
print("Total images:", len(X))