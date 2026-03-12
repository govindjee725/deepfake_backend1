import cv2
import os

DATASET_PATH = "dataset"
OUTPUT_PATH = "frames"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def extract_frames(video_path, label):

    print("Processing:", video_path)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224,224))

        filename = f"{label}_{count}.jpg"
        save_path = os.path.join(OUTPUT_PATH, filename)

        cv2.imwrite(save_path, frame)

        count += 1

    cap.release()

    print("Saved", count, "frames")


for label in ["real","fake"]:

    folder = os.path.join(DATASET_PATH,label)

    print("Checking folder:", folder)

    for video in os.listdir(folder):

        path = os.path.join(folder,video)

        extract_frames(path,label)