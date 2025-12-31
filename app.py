import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from huggingface_hub import hf_hub_download
import keras
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Breast Cancer AI Screener", page_icon="🌸", layout="wide")

st.title("🌸 Mega Breast Cancer Screening App")
st.markdown("AI trained on CBIS-DDSM dataset – Upload mammogram to get instant prediction")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
    return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ AI Model Loaded Successfully!")

uploaded_file = st.file_uploader("Upload Mammogram (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Mammogram", use_column_width=True)
    
    with col2:
        img_array = np.array(image.convert("RGB"))
        img_array = cv2.resize(img_array, (50, 50))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("AI is analyzing the image..."):
            prediction = model.predict(img_array)[0]
            cancer_prob = prediction[0]
        
        st.subheader("🔍 AI Prediction")
        if cancer_prob >= 0.5:
            st.error(f"**High Risk - Malignant** (Probability: {cancer_prob:.1%})")
            st.write("Please consult a doctor as soon as possible!")
        else:
            st.success(f"**Low Risk - Benign** (Probability: {cancer_prob:.1%})")
        
        st.warning("This is an AI tool for educational purposes only – NOT a medical diagnosis.")

st.markdown("## 🎗️ Breast Cancer Prevention & Risk Factors")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Main Risk Factors")
    st.write("""
    - Age (especially over 50)
    - Family history or BRCA gene mutations
    - Dense breast tissue
    - Obesity and excessive alcohol
    - Never having breastfed
    """)

with col4:
    st.subheader("How to Reduce Risk")
    st.write("""
    - Regular mammograms starting age 40–50
    - Maintain healthy weight
    - Exercise regularly
    - Limit alcohol consumption
    - Breastfeed if possible
    - Know your family history
    """)

st.info("Early detection saves lives – regular screening is key! 💪")
