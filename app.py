import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from huggingface_hub import hf_hub_download
import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Breast Cancer AI Screening Tool", page_icon="🎗️", layout="centered")

st.markdown("""
<style>
    .stApp {background: linear-gradient(to bottom, #fff5f8, #ffffff);}
    .main-header {font-size: 2.7rem; color: #C2185B; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.3rem; color: #666; text-align: center; margin-bottom: 40px;}
    .disclaimer {background-color: #ffebee; padding: 20px; border-radius: 12px; border-left: 5px solid #E91E63; margin: 30px 0;}
    .note {background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4CAF50;}
    .analysis-box {background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;}
    .footer {text-align: center; margin-top: 60px; color: #888; font-size: 0.95rem; padding: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎗️ Breast Cancer AI Screening Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Educational AI analysis • Trained on CBIS-DDSM • Demonstrates real-world AI challenges</p>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
<strong>⚠️ Critical Medical Disclaimer</strong><br><br>
This tool is <strong>strictly educational and research-oriented</strong>.<br>
The model often overestimates risk on normal dense breasts due to training on abnormality-focused data.<br>
<strong>NOT for clinical use or diagnosis</strong>. Results require verification by qualified radiologists.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='note'>
<strong>📊 Model Behavior Note</strong><br>
Normal or dense mammograms may receive elevated probabilities due to natural tissue patterns resembling abnormalities in training data.<br>
This demonstrates a common AI challenge: false positives in screening tools. Clinical context (patient history, additional views) is essential.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("Loading AI model..."):
        model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
        return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ AI Model Loaded")

def process_image(img_pil):
    img = img_pil.convert("RGB").resize((50, 50))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    return prediction[0]

# Samples
st.markdown("### 📊 Test with Your Sample Mammograms (Mini-MIAS)")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/wiseman-s/test2/main/sample%20images/"

sample_images = ["mdb215.png", "mdb216.png", "mdb217.png", "mdb218.png", "mdb219.png",
                 "mdb220.png", "mdb221.png", "mdb222.png", "mdb223.png", "mdb224.png"]

sample_labels = {
    "mdb215.png": "mdb215 – Normal (Dense breast)",
    "mdb216.png": "mdb216 – Malignant Calcification",
    # ... (keep your labels)
}

selected_filename = st.selectbox("Select sample", options=[""] + sample_images, format_func=lambda x: sample_labels.get(x, x))

selected_image = None
if selected_filename:
    url = GITHUB_RAW_BASE + selected_filename
    try:
        response = requests.get(url)
        selected_image = Image.open(BytesIO(response.content))
        st.image(selected_image, caption=sample_labels[selected_filename], use_column_width=True)
    except:
        st.error("Image load failed")

# Upload
st.markdown("### 📤 Or Upload Your Own")
uploaded_file = st.file_uploader("Upload mammogram", type=["jpg", "png", "jpeg"])

if uploaded_file or selected_image:
    image = Image.open(uploaded_file) if uploaded_file else selected_image
    source = "Uploaded" if uploaded_file else "Sample"

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(image, caption=source, use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 AI Analysis Result")
        with st.spinner("Analyzing..."):
            prob = process_image(image)
        
        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
        st.markdown(f"**Raw Malignancy Probability: {prob:.1%}**")
        
        if prob >= 0.9:
            st.error("**HIGH RISK ASSESSMENT**")
            st.markdown("Strong suspicious features detected – urgent clinical review recommended")
        elif prob >= 0.7:
            st.warning("**MODERATE TO HIGH RISK**")
            st.markdown("Elevated probability – may indicate abnormality or dense tissue overlap. Further evaluation needed")
        else:
            st.success("**LOW RISK ASSESSMENT**")
            st.markdown("Lower probability – consistent with normal findings, though dense tissue can elevate scores")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Prevention & Footer
st.markdown("## 🎗️ Prevention & Awareness")
st.write("Start screening early • Healthy lifestyle • Know your family history")

st.markdown("""
<div class='footer'>
    <strong>System by Simon</strong> • Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a><br>
    © 2025 Breast Cancer AI Educational Tool
</div>
""", unsafe_allow_html=True)
