import os
import cv2
import numpy as np

FRAME_PATH = "frames"

data = []
labels = []

for label_name in os.listdir(FRAME_PATH):

    folder_path = os.path.join(FRAME_PATH, label_name)

    for img in os.listdir(folder_path):

        path = os.path.join(folder_path, img)

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(image, (48,48))
        image = image / 255.0

        data.append(image)

        if label_name == "fake":
            labels.append(1)
        else:
            labels.append(0)

X = np.array(data)
y = np.array(labels)

print("Dataset Loaded")
print("Total images:", len(X))