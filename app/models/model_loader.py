from tensorflow.keras.models import load_model

model = load_model("saved_models/deepfake_model.h5")

def get_model():
    return model