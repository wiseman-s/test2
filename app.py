import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Must be tensorflow

from huggingface_hub import hf_hub_download
import keras
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Mega Breast Cancer AI", page_icon="🌸", layout="wide")

st.title("🌸 Mega Breast Cancer Screening & Prevention App")
st.markdown("**AI Model trained on CBIS-DDSM dataset** – Upload mammogram for instant Benign/Malignant prediction")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
    return keras.saving.load_model(model_path)

model = load_model()
st.success("AI Model Loaded Successfully!")

uploaded_file = st.file_uploader("Upload Mammogram Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Mammogram", use_column_width=True)
    
    with col2:
        st.write("### Processing...")
        img_array = np.array(image.convert("RGB"))  # RGB channels
        img_array = cv2.resize(img_array, (50, 50))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch
        
        with st.spinner("AI Predicting..."):
            prediction = model.predict(img_array)[0]
            cancer_prob = prediction[0]  # Class 0 = cancer probability (from original model)
        
        st.write("### 🔍 AI Result")
        if cancer_prob >= 0.5:
            st.error(f"**High Risk (Malignant/Cancerous)** – Probability: {cancer_prob:.1%}")
            st.write("Please consult a radiologist/doctor immediately!")
        else:
            st.success(f"**Low Risk (Benign)** – Probability: {cancer_prob:.1%}")
        
        st.warning("⚠️ This is AI assistance for educational purposes – NOT a substitute for professional medical diagnosis.")

# Causes & Prevention Section
st.markdown("## 🎗️ Causes, Risk Factors & Prevention")
col3, col4 = st.columns(2)
with col3:
    st.write("### Major Risk Factors")
    st.image("https://cdn.cancercenter.com/-/media/ctca/images/feature-block-images/medical-illustrations/coh-breast-cancer-risk-factors-dtm.jpg", use_column_width=True)
    st.image("https://www.shutterstock.com/shutterstock/photos/2052977840/display_1500/stock-vector-breast-cancer-risk-factors-infographic-vector-illustration-2052977840.jpg", use_column_width=True)

with col4:
    st.write("### Prevention Tips")
    st.image("https://www.shutterstock.com/shutterstock/photos/2439215303/display_1500/stock-vector-breast-cancer-awareness-infographic-vector-illustration-with-prevention-tips-statistics-and-hope-2439215303.jpg", use_column_width=True)
    st.image("https://www.shutterstock.com/shutterstock/photos/1797900529/display_1500/stock-vector-set-of-breast-cancer-awareness-prevention-tips-healthcare-infographic-vector-illustration-1797900529.jpg", use_column_width=True)

st.info("Early detection via regular screening saves lives – Start mammograms from age 40-50!")

# App inspiration
st.markdown("### Similar Medical AI Apps Look Like This")![](grok_render_searched_image_card_json={"cards":[{"cardId":"9e272f","imageId":"14","size":"LARGE"},{"cardId":"148674","imageId":"15","size":"LARGE"},{"cardId":"c215f7","imageId":"16","size":"LARGE"}]})

### How to Deploy Live (Free)
1. Save code kama `app.py`
2. Create GitHub repo → add app.py + requirements.txt (content: streamlit\nkeras\ntensorflow\nhuggingface_hub\nopencv-python\npillow\nnumpy)
3. Go to **streamlit.io/cloud** → New app → Connect GitHub → Deploy!
   - Link itakua kama: yourname-breast-cancer-app.streamlit.app

Mzee, hii ni **mega project complete** – ina screening AI + education section + visuals! Unaweza share na friends, submit conferences, au expand (add risk calculator).  
Uniambie kama unataka tu-add more features au deploy help – tuko pamoja hadi international level! 🔥🚀🌟 Kazi njema sana bro!
