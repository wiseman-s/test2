import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="Mega Breast Cancer AI", page_icon="🌸", layout="wide")

st.title("🌸 Mega Breast Cancer Screening & Prevention App")
st.markdown("**AI trained on CBIS-DDSM dataset** – Test with samples or upload your mammogram")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
    return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ AI Model Loaded Successfully!")

# Function to process image (upload or URL)
def process_image(img_pil):
    img = img_pil.convert("RGB")
    img = img.resize((50, 50))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    cancer_prob = prediction[0]  # Class 0 = cancer probability
    
    return cancer_prob

# Sample images URLs (reliable medical examples)
sample_images = {
    "Benign Sample 1 (Normal)": "https://www.mtmi.net/sites/default/files/styles/large/public/bi-rads_category_0_example_mammogram_-_mtmi.png?itok=60AOGeNB",
    "Benign Sample 2": "https://www.mtmi.net/sites/default/files/styles/large/public/bi-rads_category_2_example_mammogram_-_mtmi.png?itok=-TjL6HN2",
    "Malignant Sample 1 (BI-RADS 5)": "https://www.mtmi.net/sites/default/files/styles/large/public/bi-rads_category_5_example_mammogram_-_mtmi.png?itok=Rz548m55",
    "Malignant Sample 2": "https://radiologyassistant.nl/img/containers/main/bi-rads-for-mammography-and-ultrasound-2013/a53de98dfa31d0_10b-composition.jpg/ebd0bbed9a2468d1ec385fe9fa45288c.jpg"
}

st.markdown("### 🧪 Quick Test with Sample Mammograms")
cols = st.columns(len(sample_images))
for idx, (label, url) in enumerate(sample_images.items()):
    with cols[idx]:
        if st.button(label):
            response = requests.get(url)
            sample_img = Image.open(BytesIO(response.content))
            st.image(sample_img, caption=label, use_column_width=True)
            
            with st.spinner("AI Analyzing..."):
                prob = process_image(sample_img)
            
            if prob >= 0.5:
                st.error(f"**High Risk (Malignant)** – Probability: {prob:.1%}")
            else:
                st.success(f"**Low Risk (Benign)** – Probability: {prob:.1%}")

# Upload section
st.markdown("### 📤 Or Upload Your Own Mammogram")
uploaded_file = st.file_uploader("Choose JPG/PNG/JPEG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Mammogram", use_column_width=True)
    
    with st.spinner("AI Analyzing..."):
        prob = process_image(image)
    
    st.subheader("🔍 AI Prediction Result")
    if prob >= 0.5:
        st.error(f"**High Risk (Malignant/Cancerous)** – Probability: {prob:.1%}\n\nConsult a doctor urgently!")
    else:
        st.success(f"**Low Risk (Benign)** – Probability: {prob:.1%}")
    
    st.warning("⚠️ Educational/research tool only – NOT a medical diagnosis.")

# Prevention & Infographics
st.markdown("## 🎗️ Risk Factors & Prevention")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Major Risk Factors")
    st.image("https://cdn.cancercenter.com/-/media/ctca/images/feature-block-images/medical-illustrations/coh-breast-cancer-risk-factors-dtm.jpg", use_column_width=True)
    st.image("https://www.shutterstock.com/shutterstock/photos/2052977840/display_1500/stock-vector-breast-cancer-risk-factors-infographic-vector-illustration-2052977840.jpg", use_column_width=True)

with col2:
    st.subheader("Prevention Tips")
    st.image("https://www.shutterstock.com/shutterstock/photos/2439215303/display_1500/stock-vector-breast-cancer-awareness-infographic-vector-illustration-with-prevention-tips-statistics-and-hope-2439215303.jpg", use_column_width=True)
    st.image("https://www.shutterstock.com/shutterstock/photos/1797900529/display_1500/stock-vector-set-of-breast-cancer-awareness-prevention-tips-healthcare-infographic-vector-illustration-1797900529.jpg", use_column_width=True)

st.info("Early detection saves lives – Regular screening from age 40+ is key! 💪")
