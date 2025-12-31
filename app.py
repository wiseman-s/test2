import streamlit as st
import os
import torch
from transformers import AutoModel
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Breast Cancer AI Screening Tool", page_icon="🎗️", layout="centered")

# Professional styling (soft pink theme)
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

st.markdown("<h1 class='main-header'>🎗️ Breast Cancer AI Screening Assistant (Upgraded 2025)</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>State-of-the-art AI analysis • Powered by MammoScreen model • Educational & research tool</p>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
<strong>⚠️ Important Medical Disclaimer</strong><br><br>
This upgraded AI tool uses advanced deep learning for preliminary educational analysis. 
It is <strong>not a diagnostic device</strong> and results require confirmation by qualified radiologists. 
Early detection through clinical screening remains essential.
</div>
""", unsafe_allow_html=True)

# Load models (crop + classification)
@st.cache_resource
def load_models():
    with st.spinner("Loading advanced AI models... (this may take 1-2 minutes first time)"):
        crop_model = AutoModel.from_pretrained("ianpan/mammo-crop", trust_remote_code=True)
        class_model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
        return crop_model.eval(), class_model.eval()

crop_model, class_model = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"
crop_model = crop_model.to(device)
class_model = class_model.to(device)

st.success("✅ Advanced AI Models Loaded (MammoScreen 2025)")

def preprocess_and_predict(img_pil):
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    
    # Crop breast area
    img_shape = torch.tensor([img_cv.shape[:2]]).to(device)
    x = crop_model.preprocess(img_cv)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        coords = crop_model(x, img_shape)[0].cpu().numpy()
    x1, y1, w, h = coords
    cropped = img_cv[int(y1):int(y1+h), int(x1):int(x1+w)]
    
    # Classify
    x = class_model.preprocess(cropped)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = class_model(x).sigmoid().item()
    
    return cropped, pred

# Sample Images Section
st.markdown("### 📊 Test with Built-in Sample Mammograms (Mini-MIAS Dataset)")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/wiseman-s/test2/main/sample%20images/"

sample_images = ["mdb215.png", "mdb216.png", "mdb217.png", "mdb218.png", "mdb219.png",
                 "mdb220.png", "mdb221.png", "mdb222.png", "mdb223.png", "mdb224.png"]

sample_labels = { ... }  # Same as before

selected_filename = st.selectbox("Select sample", options=[""] + sample_images, format_func=lambda x: sample_labels.get(x, x))

selected_image = None
if selected_filename:
    url = GITHUB_RAW_BASE + selected_filename
    try:
        response = requests.get(url)
        selected_image = Image.open(BytesIO(response.content))
        st.image(selected_image, caption=sample_labels[selected_filename])
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
        st.image(image, caption=source)
    
    with col2:
        st.markdown("### 🔍 AI Analysis Result (Upgraded Model)")
        with st.spinner("Advanced processing..."):
            cropped, prob = preprocess_and_predict(image)
        
        st.image(cropped, caption="Cropped Breast Region")
        
        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
        st.markdown(f"**Malignancy Probability: {prob:.1%}**")
        
        if prob >= 0.7:
            st.error("**HIGH RISK**")
            st.markdown("Strong suspicious features detected – urgent clinical review recommended")
        elif prob >= 0.4:
            st.warning("**INTERMEDIATE RISK**")
            st.markdown("Moderate suspicious features – further imaging advised")
        else:
            st.success("**LOW RISK**")
            st.markdown("No highly suspicious features – continue routine screening")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Prevention & Footer same as before

st.markdown("""
<div class='footer'>
    <strong>System by Simon</strong> • Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a><br>
    © 2025 Upgraded Breast Cancer AI • Powered by MammoScreen • Global Awareness Tool
</div>
""", unsafe_allow_html=True)
