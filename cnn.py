import numpy as np
import cv2
import streamlit as st


@st.cache_resource(show_spinner="Memuat model MobileNetV2...")
def load_model():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet")


def _prepare(arr):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    x = cv2.resize(arr, (224, 224)).astype(np.float32)
    return preprocess_input(np.expand_dims(x, 0))


def classify(arr, top=5):
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    model = load_model()
    preds = model.predict(_prepare(arr), verbose=0)
    return [(label.replace("_", " ").title(), float(score))
            for _, label, score in decode_predictions(preds, top=top)[0]]


def gradcam(arr):
    """Mengembalikan gambar dengan peta panas area yang paling memengaruhi prediksi."""
    import tensorflow as tf
    model = load_model()
    x = tf.convert_to_tensor(_prepare(arr))
    grad_model = tf.keras.models.Model(model.inputs,
                                       [model.get_layer("Conv_1").output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(x)
        score = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (arr.shape[1], arr.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(np.ascontiguousarray(arr), 0.55, heat, 0.45, 0)