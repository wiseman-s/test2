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

# Professional styling
st.markdown("""
<style>
    .stApp {background: linear-gradient(to bottom, #fff5f8, #ffffff);}
    .main-header {font-size: 2.7rem; color: #C2185B; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.3rem; color: #666; text-align: center; margin-bottom: 40px;}
    .disclaimer {background-color: #ffebee; padding: 20px; border-radius: 12px; border-left: 5px solid #E91E63; margin: 30px 0;}
    .analysis-box {background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;}
    .footer {text-align: center; margin-top: 60px; color: #888; font-size: 0.95rem; padding: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎗️ Breast Cancer AI Screening Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Preliminary AI analysis • Trained on CBIS-DDSM • Educational tool only</p>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
<strong>⚠️ Important Medical Disclaimer</strong><br><br>
This AI provides educational analysis only. Model may overestimate risk on dense normal breasts due to training limitations.<br>
<strong>NOT a substitute for professional diagnosis</strong>. Always consult qualified radiologists.
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
    return prediction[0]  # Cancer probability

# Your Sample Images
st.markdown("### 📊 Test with Your Sample Mammograms (Mini-MIAS)")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/wiseman-s/test2/main/sample%20images/"

sample_images = ["mdb215.png", "mdb216.png", "mdb217.png", "mdb218.png", "mdb219.png",
                 "mdb220.png", "mdb221.png", "mdb222.png", "mdb223.png", "mdb224.png"]

sample_labels = {
    "mdb215.png": "mdb215 – Normal (Dense breast)",
    "mdb216.png": "mdb216 – Malignant Calcification",
    "mdb217.png": "mdb217 – Normal",
    "mdb218.png": "mdb218 – Benign Calcification",
    "mdb219.png": "mdb219 – Benign Calcification",
    "mdb220.png": "mdb220 – Normal",
    "mdb221.png": "mdb221 – Normal (Dense)",
    "mdb222.png": "mdb222 – Benign Calcification",
    "mdb223.png": "mdb223 – Benign Calcification",
    "mdb224.png": "mdb224 – Normal"
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
        
        # Adjusted thresholds to reduce false positives
        if prob >= 0.8:
            st.error("**HIGH RISK**")
            st.markdown("Strong suspicious features – urgent clinical review recommended")
        elif prob >= 0.5:
            st.warning("**MODERATE RISK**")
            st.markdown("Some patterns detected – may be dense tissue or early changes. Clinical correlation needed")
        else:
            st.success("**LOW RISK**")
            st.markdown("No highly suspicious features – consistent with normal/benign findings")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Prevention
st.markdown("## 🎗️ Prevention & Awareness")
st.image("https://www.iarc.who.int/wp-content/uploads/2023/10/BCAM_2_zoom.jpg", caption="Global Statistics")
st.write("- Annual screening from age 40\n- Healthy lifestyle\n- Know your risk")

# Footer
st.markdown("""
<div class='footer'>
    <strong>System by Simon</strong> • Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a><br>
    © 2025 Breast Cancer AI Tool • Educational Platform
</div>
""", unsafe_allow_html=True)
