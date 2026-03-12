import numpy as np
import cv2
import tensorflow as tf


def generate_heatmap(model, image):

    img = image / 255.0
    img = np.expand_dims(img, axis=0)

    img_tensor = tf.convert_to_tensor(img)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)

    grads = tape.gradient(prediction, img_tensor)[0]

    heatmap = tf.reduce_mean(tf.abs(grads), axis=-1)

    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(frame, heatmap):

    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return result