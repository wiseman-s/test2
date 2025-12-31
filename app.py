import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras import backend as K
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from io import BytesIO
import cv2
import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIG (CLEAN MEDICAL UI)
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer AI Screening Tool",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.analysis-box {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 20px;
    background-color: #fafafa;
}
.footer {
    text-align: center;
    color: #666;
    margin-top: 40px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🩺 Breast Cancer AI Screening Tool")
st.caption("Educational & Research Use Only")

st.warning(
    "⚠️ **Medical Disclaimer**\n\n"
    "This system is strictly for educational and research purposes.\n\n"
    "The underlying AI model was trained primarily on lesion-centered datasets "
    "(CBIS-DDSM style) and is **not validated for population screening**.\n\n"
    "Normal or dense breasts may be flagged due to tissue patterns resembling abnormalities.\n\n"
    "**This tool must NOT be used for diagnosis or clinical decision-making.**"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    with st.spinner("Loading AI model..."):
        model_path = hf_hub_download(
            repo_id="maiurilorenzo/CBIS-DDSM-CNN",
            filename="CNN_model.h5"
        )
        return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ Model loaded successfully")

# --------------------------------------------------
# IMAGE PREPROCESSING (CORRECTED)
# --------------------------------------------------
def preprocess_image(img_pil):
    img = img_pil.convert("L")
    img = img.resize((224, 224))

    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

# --------------------------------------------------
# GRAD-CAM FUNCTIONS
# --------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_array)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-7
    return heatmap.numpy()

def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_rgb = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
    return overlay

# --------------------------------------------------
# SAMPLE IMAGES
# --------------------------------------------------
st.subheader("📊 Sample Mammograms")

BASE_URL = "https://raw.githubusercontent.com/wiseman-s/test2/main/sample%20images/"
samples = ["mdb215.png", "mdb216.png", "mdb217.png", "mdb218.png", "mdb219.png"]

labels = {
    "mdb215.png": "Normal (Dense)",
    "mdb216.png": "Malignant Calcification",
    "mdb217.png": "Benign Mass",
    "mdb218.png": "Normal",
    "mdb219.png": "Benign"
}

selected = st.selectbox(
    "Select sample image",
    options=[""] + samples,
    format_func=lambda x: labels.get(x, x)
)

selected_image = None
if selected:
    try:
        r = requests.get(BASE_URL + selected)
        selected_image = Image.open(BytesIO(r.content))
        st.image(selected_image, caption=labels[selected], use_column_width=True)
    except:
        st.error("Failed to load image")

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
st.subheader("📤 Upload Mammogram")
uploaded = st.file_uploader("Upload PNG / JPG image", type=["png", "jpg", "jpeg"])

image = Image.open(uploaded) if uploaded else selected_image

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if image:
    st.subheader("🔍 AI Analysis")

    img_array = preprocess_image(image)
    raw_score = float(model.predict(img_array, verbose=0)[0][0])

    # Dataset bias calibration
    calibrated_score = max(0.0, raw_score - 0.25)

    st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
    st.write(f"**Raw abnormality similarity score:** {raw_score:.3f}")
    st.write(f"**Bias-calibrated score:** {calibrated_score:.3f}")

    if calibrated_score >= 0.97:
        st.error("High abnormality similarity detected")
    elif calibrated_score >= 0.85:
        st.warning("Moderate abnormality similarity (dense tissue overlap possible)")
    else:
        st.success("Low abnormality similarity")

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # GRAD-CAM VISUALIZATION
    # --------------------------------------------------
    st.subheader("🧠 Model Attention Visualization (Grad-CAM)")

    try:
        LAST_CONV_LAYER = "conv2d_3"  # adjust if model differs
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        overlay = overlay_heatmap(heatmap, image)

        st.image(
            overlay,
            caption="Highlighted regions influenced the model's assessment",
            use_column_width=True
        )

        st.caption(
            "Bright regions indicate areas the CNN focused on. "
            "In normal dense breasts, glandular tissue often triggers elevated responses."
        )

    except Exception:
        st.info("Grad-CAM not available for this model architecture")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
<strong>Breast Cancer AI Screening Tool</strong><br>
Developed by Simon • 2025<br>
Educational & Research Use Only
</div>
""", unsafe_allow_html=True)
