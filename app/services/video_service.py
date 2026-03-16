import cv2
from mtcnn import MTCNN

detector = MTCNN()

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # take every 5th frame
        if count % 5 == 0:

            faces = detector.detect_faces(frame)

            if faces:
                x, y, w, h = faces[0]['box']

                # crop face
                face = frame[y:y+h, x:x+w]

                if face.size != 0:
                    face = cv2.resize(face, (224,224))
                    frames.append(face)

        count += 1

    cap.release()

    return frames