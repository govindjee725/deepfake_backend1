from app.services.video_service import extract_frames
from app.utils.heatmap import generate_heatmap, overlay_heatmap
from app.models.model_loader import get_model
import numpy as np
import cv2
import os

# def detect_deepfake(video_path):

#     model = get_model()   


def clear_frames_folder():
    frames_dir = "frames"

    if os.path.exists(frames_dir):
        for file in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, file)
            try:
                os.remove(file_path)
            except:
                pass


def detect_deepfake(video_path):

    model = get_model()

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {
            "deepfake_score": 0,
            "result": "No Frames Detected",
            "heatmap_image": None
        }

    scores = []
    heatmap_path = None

    os.makedirs("outputs", exist_ok=True)

    for i, frame in enumerate(frames[::10]):

        input_frame = frame / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        if input_frame.shape != (1,224,224,3):
            continue

        prediction = model.predict(input_frame, verbose=0)

        score = float(prediction[0][0])
        scores.append(score)

        if i == 0:
            heatmap = generate_heatmap(model, frame)
            visual = overlay_heatmap(frame, heatmap)
            heatmap_path = "outputs/heatmap_result.jpg"
            cv2.imwrite(heatmap_path, visual)

    if len(scores) == 0:
        return {
            "deepfake_score": 0,
            "result": "Prediction Failed",
            "heatmap_image": None
        }

    avg_score = sum(scores) / len(scores)

    result = "Fake" if avg_score > 0.5 else "Real"
    # delete uploaded video
    if os.path.exists(video_path):
        os.remove(video_path)

    # delete extracted frames
    clear_frames_folder()

    return {
        "deepfake_score": round(float(avg_score), 3),
        "result": result,
        "heatmap_image": heatmap_path
    }
    # 🔥 CLEANUP SECTION
