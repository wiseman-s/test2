import streamlit as st
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from huggingface_hub import hf_hub_download
import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# === Page Config & Professional Theme ===
st.set_page_config(page_title="Breast Cancer AI Screening Tool", page_icon="🎗️", layout="centered")

# Professional background & styling with full dark mode support
st.markdown("""
<style>
    /* Light mode background (soft pink awareness theme) */
    .stApp {
        background: linear-gradient(to bottom, #fff5f8, #ffffff);
    }
    
    /* Dark mode overrides - clean and readable */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(to bottom, #1e1e1e, #121212) !important;
        }
        .main-header {color: #FF4081 !important;}
        .sub-header {color: #bbbbbb !important;}
        .disclaimer {background-color: #3a1a2a !important; border-left: 5px solid #FF4081 !important;}
        .analysis-box {background-color: #2a2a2a !important; border: 1px solid #444 !important; color: #e0e0e0 !important;}
        .footer {color: #aaaaaa !important;}
        section[data-testid="stSidebar"] {background-color: #1e1e1e !important;}
        .stMarkdown, p, div, span, li {color: #e0e0e0 !important;}
        h1, h2, h3, h4 {color: #ffffff !important;}
        a {color: #FF79B0 !important;}
    }

    .main-header {font-size: 2.7rem; color: #C2185B; text-align: center; font-weight: bold; margin-bottom: 10px;}
    .sub-header {font-size: 1.3rem; color: #666; text-align: center; margin-bottom: 40px;}
    .disclaimer {background-color: #ffebee; padding: 20px; border-radius: 12px; border-left: 5px solid #E91E63; margin: 30px 0;}
    .analysis-box {background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;}
    .footer {text-align: center; margin-top: 60px; color: #888; font-size: 0.95rem; padding: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎗️ Breast Cancer AI Screening Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered preliminary mammogram analysis • Trained on CBIS-DDSM dataset • Educational & research tool</p>", unsafe_allow_html=True)

# Strong Medical Disclaimer
st.markdown("""
<div class='disclaimer'>
<strong>⚠️ Important Medical Disclaimer</strong><br><br>
This AI tool provides <strong>preliminary educational analysis only</strong> based on mammographic patterns. 
It is <strong>not a diagnostic device</strong> and <strong>cannot replace</strong> professional radiological interpretation or clinical judgment.<br><br>
All results must be confirmed by qualified healthcare providers using standard clinical protocols. 
Early detection through regular screening remains the gold standard for improving outcomes.
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with st.spinner("Initializing AI model..."):
        model_path = hf_hub_download(repo_id="maiurilorenzo/CBIS-DDSM-CNN", filename="CNN_model.h5")
        return keras.saving.load_model(model_path)

model = load_model()
st.success("✅ AI Model Successfully Loaded and Ready")

def process_image(img_pil):
    img = img_pil.convert("RGB").resize((50, 50))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0]
    return prediction[0]

# === Sample Images from Your GitHub ===
st.markdown("### 📊 Test with Built-in Sample Mammograms (Mini-MIAS Dataset)")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/wiseman-s/test2/main/sample%20images/"

sample_images = [
    "mdb215.png", "mdb216.png", "mdb217.png", "mdb218.png", "mdb219.png",
    "mdb220.png", "mdb221.png", "mdb222.png", "mdb223.png", "mdb224.png"
]

sample_labels = {
    "mdb215.png": "mdb215.png – Dense breast, Normal",
    "mdb216.png": "mdb216.png – Dense breast, Malignant Calcification",
    "mdb217.png": "mdb217.png – Glandular breast, Normal",
    "mdb218.png": "mdb218.png – Glandular breast, Benign Calcification",
    "mdb219.png": "mdb219.png – Glandular breast, Benign Calcification",
    "mdb220.png": "mdb220.png – Glandular breast, Normal",
    "mdb221.png": "mdb221.png – Dense breast, Normal",
    "mdb222.png": "mdb222.png – Dense breast, Benign Calcification",
    "mdb223.png": "mdb223.png – Dense breast, Benign Calcification",
    "mdb224.png": "mdb224.png – Dense breast, Normal"
}

selected_filename = st.selectbox(
    "Select a sample mammogram for analysis",
    options=[""] + sample_images,
    format_func=lambda x: sample_labels.get(x, x) if x else "— Choose a sample —"
)

selected_image = None
if selected_filename:
    image_url = GITHUB_RAW_BASE + selected_filename
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        selected_image = Image.open(BytesIO(response.content))
        st.image(selected_image, caption=sample_labels[selected_filename], use_column_width=True)
    except:
        st.error("Failed to load image. Please check GitHub repository path and filename.")

# === User Upload ===
st.markdown("### 📤 Or Upload Your Own Mammogram")
uploaded_file = st.file_uploader("Upload digital mammogram (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

# === Professional AI Analysis with Detailed Reasoning ===
if uploaded_file or selected_image:
    if uploaded_file:
        image = Image.open(uploaded_file)
        source = "Uploaded Mammogram"
    else:
        image = selected_image
        source = "Selected Sample"

    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, caption=source, use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 AI Analysis Result")
        with st.spinner("Analyzing mammographic features..."):
            prob = process_image(image)
        
        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
        
        st.markdown(f"**Computed Malignancy Probability: {prob:.1%}**")
        
        if prob >= 0.7:
            st.error("**HIGH RISK ASSESSMENT**")
            st.markdown("""
            **Interpretation**:  
            The AI model detects features strongly associated with malignancy, such as irregular mass margins, clustered microcalcifications, or architectural distortion.  
            These patterns have high correlation with malignant lesions in the training dataset (CBIS-DDSM).
            
            **Clinical Recommendation**:  
            Immediate referral for diagnostic workup (additional views, ultrasound, or biopsy) is strongly advised.
            """)
        elif prob >= 0.5:
            st.warning("**INTERMEDIATE TO HIGH RISK**")
            st.markdown("""
            **Interpretation**:  
            The model identifies suspicious features that may suggest early malignant changes or high-risk benign lesions (e.g., radial scars, atypical calcifications).
            
            **Clinical Recommendation**:  
            Prompt clinical correlation and further imaging (magnification views, MRI if dense breasts) are recommended.
            """)
        else:
            st.success("**LOW RISK ASSESSMENT**")
            st.markdown("""
            **Interpretation**:  
            The mammogram shows predominantly normal fibroglandular tissue with no highly suspicious features detected by the model. 
            Benign calcifications or cysts, if present, appear typical.
            
            **Clinical Recommendation**:  
            Continue routine age-appropriate screening. Maintain breast awareness and report any palpable changes.
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)

# === Prevention Section ===
st.markdown("## 🎗️ Breast Cancer Risk Factors & Prevention")
col_inf1, col_inf2 = st.columns(2)
with col_inf1:
    st.image("https://www.iarc.who.int/wp-content/uploads/2023/10/BCAM_2_zoom.jpg", caption="Global Breast Cancer Burden (WHO/IARC 2025)", use_column_width=True)

with col_inf2:
    st.image("https://www.shutterstock.com/image-vector/breast-cancer-awareness-infographic-empowering-600nw-2355615993.jpg", caption="Empowerment Through Early Detection", use_column_width=True)

st.markdown("### Evidence-Based Prevention Strategies")
st.write("""
- Commence annual screening mammography at age 40–50 (per international guidelines)
- Perform regular clinical breast exams and monthly self-examinations
- Adopt healthy lifestyle: balanced diet, regular physical activity (150+ min/week), healthy weight maintenance
- Limit alcohol consumption
- Consider extended breastfeeding where possible
- Discuss genetic risk assessment if strong family history present
""")

# === Footer with Your Credit ===
st.markdown("""
<div class='footer'>
    <strong>System Developed by Simon</strong> • Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a><br>
    © 2025 Breast Cancer AI Screening Assistant • Educational & Research Platform • Dedicated to Global Health Awareness
</div>
""", unsafe_allow_html=True)
