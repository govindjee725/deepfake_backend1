# import cv2
# import os
# from mtcnn import MTCNN

# DATASET_PATH = "dataset"
# OUTPUT_PATH = "frames"

# detector = MTCNN()

# os.makedirs(os.path.join(OUTPUT_PATH, "real"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, "fake"), exist_ok=True)


# def extract_frames(video_path, label):

#     print("Processing:", video_path)

#     cap = cv2.VideoCapture(video_path)

#     count = 0
#     saved = 0

#     video_name = os.path.splitext(os.path.basename(video_path))[0]

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         if count % 5 == 0:

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             faces = detector.detect_faces(rgb)

#             if len(faces) > 0:

#                 x, y, w, h = faces[0]["box"]

#                 face = rgb[y:y+h, x:x+w]

#                 face = cv2.resize(face, (224,224))

#                 filename = f"{label}_{video_name}_{saved}.jpg"

#                 save_path = os.path.join(OUTPUT_PATH, label, filename)

#                 cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

#                 saved += 1

#         count += 1

#     cap.release()

#     print(f"Saved {saved} frames from {video_path}")


# for label in ["real", "fake"]:

#     folder = os.path.join(DATASET_PATH, label)

#     print("Checking folder:", folder)

#     for video in os.listdir(folder):

#         path = os.path.join(folder, video)

#         extract_frames(path, label)

import cv2
import os
from mtcnn import MTCNN

DATASET_PATH = "dataset"
OUTPUT_PATH = "frames"

detector = MTCNN()

# create output folders
for split in ["train", "val"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(OUTPUT_PATH, split, label), exist_ok=True)


def extract_frames(video_path, label, split):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(OUTPUT_PATH, split, label)

    # -------- Skip already processed videos --------
    existing = [f for f in os.listdir(output_folder) if video_name in f]

    if len(existing) >= 30:
        print("Skipping already processed:", video_name)
        return
    # ------------------------------------------------

    print("Processing:", video_path)

    cap = cv2.VideoCapture(video_path)

    count = 0
    saved = len(existing)  # continue from existing frame count

    while True:

        if saved >= 30:   # max 30 frames per video
            break

        ret, frame = cap.read()

        if not ret:
            break

        if count % 5 == 0:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = detector.detect_faces(rgb)

            if len(faces) > 0:

                x, y, w, h = faces[0]["box"]

                face = rgb[y:y+h, x:x+w]

                face = cv2.resize(face, (224,224))

                filename = f"{label}_{video_name}_{saved}.jpg"

                save_path = os.path.join(output_folder, filename)

                cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                saved += 1

        count += 1

    cap.release()

    print(f"Saved {saved} frames from {video_path}")


# loop through train and validation videos
for split in ["train", "val"]:

    for label in ["real", "fake"]:

        folder = os.path.join(DATASET_PATH, split, label)

        print("Checking folder:", folder)

        for video in os.listdir(folder):

            path = os.path.join(folder, video)

            extract_frames(path, label, split)