import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="Breast Cancer AI Screening Tool", page_icon="🎗️", layout="centered")

# Header with professional look
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #E91E63; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 30px;}
    .disclaimer {background-color: #f8d7da; padding: 15px; border-radius: 10px; margin: 20px 0;}
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
    with st.spinner("Loading AI model..."):
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

# Sample mammograms with professional examples
st.markdown("### 📊 Example Analysis (Click to Test)")
cols = st.columns(4)
sample_data = [
    ("Normal (BI-RADS 1)", "https://www.mtmi.net/sites/default/files/styles/large/public/bi-rads_category_0_example_mammogram_-_mtmi.png?itok=60AOGeNB", "Expected: Low Risk"),
    ("Benign Finding", "https://www.mtmi.net/sites/default/files/styles/large/public/bi-rads_category_2_example_mammogram_-_mtmi.png?itok=-TjL6HN2", "Expected: Low Risk"),
    ("Suspicious (BI-RADS 4)", "https://radiologybusiness.com/sites/default/files/styles/gallery/public/2022-03/comparison_2d_mammo_vs_dbt_3d_mammo_ucsf.jpg.webp?h=907d7ba9&itok=5HouL2X7", "Expected: Higher Risk"),
    ("Malignant (BI-RADS 5)", "https://www.mtmi.net/sites/default/files/styles/960x504_social/public/bi-rads_category_5_example_mammogram_-_mtmi.png?h=fda98b9f&itok=xEd34XKD", "Expected: High Risk")
]

selected_sample = None
for idx, (title, url, expected) in enumerate(sample_data):
    with cols[idx]:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        if st.button(title):
            selected_sample = (img, title, expected)

# Upload section
st.markdown("### 📤 Analyze Your Own Mammogram")
uploaded_file = st.file_uploader("Upload digital mammogram (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file or selected_sample:
    if selected_sample:
        image, title, expected = selected_sample
        st.info(f"**Sample Selected**: {title} • {expected}")
    else:
        image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Mammogram Image", use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 AI Analysis Result")
        with st.spinner("Processing image..."):
            prob = process_image(image)
        
        if prob >= 0.5:
            st.error(f"**High Risk Detected** • Probability: {prob:.1%}")
            st.write("Recommendation: Urgent clinical follow-up required")
        else:
            st.success(f"**Low Risk** • Probability of malignancy: {prob:.1%}")
            st.write("Continue regular screening as recommended")

# Professional infographics section
st.markdown("## 🎗️ Breast Cancer Awareness & Prevention")
st.markdown("### Risk Factors & Prevention Strategies")

col_inf1, col_inf2 = st.columns(2)
with col_inf1:
    st.image("https://www.iarc.who.int/wp-content/uploads/2023/10/BCAM_2_zoom.jpg", caption="Global Breast Cancer Statistics (IARC/WHO)", use_column_width=True)
    st.image("https://www.shutterstock.com/image-vector/medical-vector-illustrationbreast-cancer-prevention-600nw-2350011643.jpg", caption="Prevention Guidelines", use_column_width=True)

with col_inf2:
    st.image("https://stg-uploads.slidenest.com/template_824/templateColor_858/previewImages/breast-cancer-prevention-infographic-powerpoint-google-slides-keynote-presentation-template-1.jpg", caption="Lifestyle & Screening Recommendations", use_column_width=True)
    st.image("https://www.shutterstock.com/image-vector/breast-cancer-awareness-infographic-empowering-600nw-2355615993.jpg", caption="Empowerment & Early Detection", use_column_width=True)

st.markdown("### Key Recommendations from Global Health Authorities")
st.write("""
- Begin annual mammograms at age 40–50 (per guidelines)
- Perform regular self-exams
- Maintain healthy lifestyle: exercise, balanced diet, limit alcohol
- Know your family history and consider genetic counseling if needed
""")

st.caption("© 2025 Breast Cancer AI Screening Assistant • Educational Tool • Built with ❤️ for global awareness")
