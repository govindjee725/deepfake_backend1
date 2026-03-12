import numpy as np

def preprocess_face(face):

    face = face / 255.0

    face = np.expand_dims(face, axis=0)

    return face