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

# === NEW: Load Sample Images from Your GitHub Repo ===
st.markdown("### 📊 Test with Built-in Sample Mammograms")

# CHANGE THIS to your actual GitHub repo raw URL base
# Example: if your repo is https://github.com/yourusername/breast-cancer-app
# and images are in folder /samples/
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/yourusername/your-repo-name/main/samples/"  

# Update these with your actual image filenames (case-sensitive!)
sample_images = {
    "Normal Mammogram (Expected: Low Risk)": "normal1.png",
    "Benign Calcification (Expected: Low Risk)": "benign1.png",
    "Malignant Mass (Expected: High Risk)": "malignant1.png",
    "Suspicious Finding (Expected: High Risk)": "malignant2.png",
    # Add more as you upload
}

selected_sample_name = st.selectbox("Select a sample image to analyze", options=[""] + list(sample_images.keys()))

selected_image = None
if selected_sample_name:
    filename = sample_images[selected_sample_name]
    image_url = GITHUB_RAW_BASE + filename
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        selected_image = Image.open(BytesIO(response.content))
        st.image(selected_image, caption=selected_sample_name, use_column_width=True)
    except:
        st.error(f"Could not load image: {filename}. Check filename and GitHub path.")

# === Upload Your Own ===
st.markdown("### 📤 Or Upload Your Own Mammogram")
uploaded_file = st.file_uploader("Upload digital mammogram (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

# === Analysis Section ===
if uploaded_file or selected_image:
    if uploaded_file:
        image = Image.open(uploaded_file)
        source = "Uploaded Image"
    else:
        image = selected_image
        source = "Sample Image"

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption=f"{source}", use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 AI Analysis Result")
        with st.spinner("Processing image with AI..."):
            prob = process_image(image)
        
        if prob >= 0.5:
            st.error(f"**High Risk Detected** • Malignancy Probability: {prob:.1%}")
            st.write("**Recommendation**: Urgent clinical evaluation required")
        else:
            st.success(f"**Low Risk** • Malignancy Probability: {prob:.1%}")
            st.write("**Recommendation**: Continue routine screening")

# === Awareness & Prevention ===
st.markdown("## 🎗️ Breast Cancer Awareness & Prevention")
st.markdown("### Risk Factors & Prevention Strategies")

col_inf1, col_inf2 = st.columns(2)
with col_inf1:
    st.image("https://www.iarc.who.int/wp-content/uploads/2023/10/BCAM_2_zoom.jpg", caption="Global Statistics (WHO/IARC)", use_column_width=True)

with col_inf2:
    st.image("https://www.shutterstock.com/image-vector/breast-cancer-awareness-infographic-empowering-600nw-2355615993.jpg", caption="Early Detection & Empowerment", use_column_width=True)

st.markdown("### Key Recommendations")
st.write("""
- Annual mammograms starting age 40–50
- Monthly breast self-exams
- Healthy lifestyle: exercise, balanced diet, limited alcohol
- Genetic counseling if family history present
""")

# === Footer with Contact ===
st.markdown("""
<div class='footer'>
    <strong>System by Simon</strong> • Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a><br>
    © 2025 Breast Cancer AI Screening Assistant • Educational Tool • Built with ❤️ for global health awareness
</div>
""", unsafe_allow_html=True)
