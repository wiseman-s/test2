import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from huggingface_hub import hf_hub_download
import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="Breast Cancer AI Screener", page_icon="🌸", layout="wide")

st.title("🌸 Mega Breast Cancer Screening & Prevention App")
st.markdown("AI trained on CBIS-DDSM – Upload mammogram for instant Benign/Malignant prediction")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
    return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ AI Model Loaded Successfully!")

uploaded_file = st.file_uploader("Upload Mammogram (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Mammogram", use_column_width=True)
    
    with col2:
        # Preprocess na PIL only (no cv2 needed!)
        img = image.convert("RGB")
        img = img.resize((50, 50))  # Resize direct na PIL
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        
        with st.spinner("AI Analyzing..."):
            prediction = model.predict(img_array)[0]
            cancer_prob = prediction[0]  # Class 0 = cancer
        
        st.subheader("🔍 AI Prediction Result")
        if cancer_prob >= 0.5:
            st.error(f"**High Risk (Malignant/Cancerous)** – Probability: {cancer_prob:.1%}")
            st.write("Consult a doctor urgently!")
        else:
            st.success(f"**Low Risk (Benign)** – Probability: {cancer_prob:.1%}")
        
        st.warning("⚠️ Educational tool only – NOT a medical diagnosis.")

# Prevention Section
st.markdown("## 🎗️ Breast Cancer Risk Factors & Prevention")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Risk Factors")
    st.write("- Age >50\n- Family history/BRCA genes\n- Dense breasts\n- Obesity/alcohol\n- No breastfeeding")

with col4:
    st.subheader("Prevention Tips")
    st.write("- Regular mammograms (age 40+)\n- Healthy weight & exercise\n- Limit alcohol\n- Breastfeed if possible\n- Genetic testing if high risk")

st.info("Early screening saves lives! Start today 💪")
