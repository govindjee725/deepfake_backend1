from tensorflow.keras.models import load_model

model = None

def get_model():
    global model

    if model is None:
        model = load_model("saved_models/deepfake_model.h5")

    return model