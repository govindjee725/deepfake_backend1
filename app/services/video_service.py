import cv2

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame,(224,224))

        frames.append(frame)

    cap.release()

    return frames