from app.services.video_service import extract_frames
from app.utils.heatmap import generate_heatmap, overlay_heatmap
from app.models.model_loader import get_model
import numpy as np
import cv2
import os

model = get_model()

def detect_deepfake(video_path):

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

    # process every 8th frame (better coverage than 10)
    for i, frame in enumerate(frames[::8]):

        input_frame = frame / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        prediction = model.predict(input_frame, verbose=0)

        score = float(prediction[0][0])
        scores.append(score)

        # generate heatmap for the first frame only
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

    # better classification logic
    if avg_score > 0.65:
        result = "Fake"
    elif avg_score < 0.35:
        result = "Real"
    else:
        result = "Uncertain"

    return {
        "deepfake_score": round(float(avg_score), 3),
        "result": result,
        "heatmap_image": heatmap_path
    }