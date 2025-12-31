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
    .main-header {font-size: 2.5rem; color: #E91E63; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 30px;}
    .disclaimer {background-color: #f8d7da; padding: 15px; border-radius: 10px; margin: 20px 0;}
    .footer {text-align: center; margin-top: 50px; color: #888; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎗️ Breast Cancer AI Screening Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered preliminary mammogram analysis • Trained on CBIS-DDSM dataset • For educational & research use only</p>", unsafe_allow_html=True)

# Professional disclaimer
st.markdown("""
<div class='disclaimer'>
<strong>⚠️ Important Medical Disclaimer</strong><br>
This tool uses artificial intelligence for educational and research purposes only. 
It is <strong>NOT a substitute for professional medical diagnosis</strong>. 
All results should be verified by qualified healthcare professionals. 
Early detection through clinical screening saves lives.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("Loading AI model from Hugging Face..."):
        model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
        return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ AI Model Ready for Analysis")

def process_image(img_pil):
    img = img_pil.convert("RGB").resize((50, 50))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0]
    return prediction[0]  # Cancer probability

# === Load Sample Images from Your GitHub Repo ===
st.markdown("### 📊 Test with Built-in Sample Mammograms (Mini-MIAS Dataset)")

# UPDATE THIS LINE with your actual GitHub repo details
# Example: if repo is https://github.com/simon57/breast-cancer-ai and images in /samples folder
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/YOURUSERNAME/YOUR-REPO-NAME/main/samples/"

# Your exact sample images
sample_images = {
    "mdb215.png": "mdb215.png",
    "mdb216.png": "mdb216.png",
    "mdb217.png": "mdb217.png",
    "mdb218.png": "mdb218.png",
    "mdb219.png": "mdb219.png",
    "mdb220.png": "mdb220.png",
    "mdb221.png": "mdb221.png",
    "mdb222.png": "mdb222.png",
    "mdb223.png": "mdb223.png",
    "mdb224.png": "mdb224.png"
}

# Dropdown selection
selected_filename = st.selectbox("Select a sample mammogram to analyze", options=[""] + list(sample_images.keys()))

selected_image = None
if selected_filename:
    image_url = GITHUB_RAW_BASE + selected_filename
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        selected_image = Image.open(BytesIO(response.content))
        st.image(selected_image, caption=f"Sample: {selected_filename}", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {selected_filename}. Check GitHub path and filename. Details: {str(e)}")

# === Upload Your Own ===
st.markdown("### 📤 Or Upload Your Own Mammogram")
uploaded_file = st.file_uploader("Upload digital mammogram (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

# === Analysis ===
if uploaded_file or selected_image:
    if uploaded_file:
        image = Image.open(uploaded_file)
        source = "Your Uploaded Image"
    else:
        image = selected_image
        source = "Selected Sample Image"

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption=source, use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 AI Analysis Result")
        with st.spinner("AI processing image..."):
            prob = process_image(image)
        
        if prob >= 0.5:
            st.error(f"**High Risk Detected** • Malignancy Probability: {prob:.1%}")
            st.write("**Recommendation**: Immediate clinical follow-up advised")
        else:
            st.success(f"**Low Risk** • Malignancy Probability: {prob:.1%}")
            st.write("**Recommendation**: Continue routine screening")

# === Awareness Section ===
st.markdown("## 🎗️ Breast Cancer Awareness & Prevention")
col_inf1, col_inf2 = st.columns(2)
with col_inf1:
    st.image("https://www.iarc.who.int/wp-content/uploads/2023/10/BCAM_2_zoom.jpg", caption="Global Breast Cancer Statistics (WHO/IARC)", use_column_width=True)

with col_inf2:
    st.image("https://www.shutterstock.com/image-vector/breast-cancer-awareness-infographic-empowering-600nw-2355615993.jpg", caption="Early Detection & Empowerment", use_column_width=True)

st.markdown("### Key Prevention Tips")
st.write("""
- Start annual mammograms at age 40–50
- Perform monthly breast self-exams
- Maintain healthy weight and regular exercise
- Limit alcohol consumption
- Breastfeed if possible
- Know your family history
""")

# === Footer with Your Contact ===
st.markdown("""
<div class='footer'>
    <strong>System by Simon</strong> • Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a><br>
    © 2025 Breast Cancer AI Screening Assistant • Educational & Research Tool • Built for Global Health Awareness
</div>
""", unsafe_allow_html=True)
