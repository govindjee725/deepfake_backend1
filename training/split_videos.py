
import os
import random
import shutil

SOURCE = "dataset"
TRAIN = "dataset/train"
VAL = "dataset/val"

SPLIT_RATIO = 0.8   # 80% train, 20% validation

for label in ["fake", "real"]:

    source_folder = os.path.join(SOURCE, label)

    videos = os.listdir(source_folder)
    random.shuffle(videos)

    split_index = int(len(videos) * SPLIT_RATIO)

    train_videos = videos[:split_index]
    val_videos = videos[split_index:]

    for v in train_videos:
        shutil.move(
            os.path.join(source_folder, v),
            os.path.join(TRAIN, label, v)
        )

    for v in val_videos:
        shutil.move(
            os.path.join(source_folder, v),
            os.path.join(VAL, label, v)
        )

    print(f"{label} → train: {len(train_videos)}, val: {len(val_videos)}")

print("Video split complete")

