import os

# Set Keras backend before importing Keras
os.environ["KERAS_BACKEND"] = "tensorflow"

from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import keras


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Cancer AI Screening Tool",
    page_icon="🎗️",
    layout="centered",
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
        /* Light mode */
        .stApp {
            background: linear-gradient(to bottom, #fff5f8, #ffffff);
        }

        .main-header {
            font-size: 2.7rem;
            color: #C2185B;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .sub-header {
            font-size: 1.3rem;
            color: #666;
            text-align: center;
            margin-bottom: 40px;
        }

        .disclaimer {
            background-color: #ffebee;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #E91E63;
            margin: 30px 0;
        }

        .analysis-box {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }

        .footer {
            text-align: center;
            margin-top: 60px;
            color: #888;
            font-size: 0.95rem;
            padding: 20px;
        }

        .footer a {
            color: #C2185B;
            text-decoration: none;
        }

        /* Dark mode */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background: linear-gradient(to bottom, #1e1e1e, #121212) !important;
            }

            .main-header { color: #FF4081 !important; }
            .sub-header { color: #bbbbbb !important; }

            .disclaimer {
                background-color: #3a1a2a !important;
                border-left: 5px solid #FF4081 !important;
            }

            .analysis-box {
                background-color: #2a2a2a !important;
                border: 1px solid #444 !important;
                color: #e0e0e0 !important;
            }

            .footer { color: #aaaaaa !important; }
            .footer a { color: #FF79B0 !important; }

            section[data-testid="stSidebar"] {
                background-color: #1e1e1e !important;
            }

            .stMarkdown, p, div, span, li { color: #e0e0e0 !important; }
            h1, h2, h3, h4 { color: #ffffff !important; }
            a { color: #FF79B0 !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<h1 class="main-header">🎗️ Breast Cancer AI Screening Assistant</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="sub-header">
        AI-powered preliminary mammogram analysis •
        Trained on CBIS-DDSM dataset •
        Educational &amp; research tool
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚠️ Important Medical Disclaimer</strong><br><br>
        This AI tool provides <strong>preliminary educational analysis only</strong>
        based on mammographic patterns. It is <strong>not a diagnostic device</strong>
        and <strong>cannot replace</strong> professional radiological interpretation
        or clinical judgment.<br><br>
        All results should be interpreted by qualified healthcare professionals
        using appropriate clinical protocols.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with st.spinner("Initializing AI model..."):
        model_path = hf_hub_download(
            repo_id="maiurilorenzo/CBIS-DDSM-CNN",
            filename="CNN_model.h5",
        )
        return keras.saving.load_model(model_path)


try:
    model = load_model()
    st.success("✅ AI Model Successfully Loaded and Ready")
except Exception:
    st.error("❌ The AI model could not be loaded.")
    st.caption(
        "Please check the Hugging Face model repository, "
        "model file, and installed Keras/TensorFlow versions."
    )
    st.stop()


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(img_pil):
    img = img_pil.convert("RGB").resize((50, 50))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0]
    return float(prediction[0])


# ============================================================
# BUILT-IN SAMPLE MAMMOGRAMS
# ============================================================

st.markdown("### 📊 Test with Built-in Sample Mammograms")

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/wiseman-s/test2/main/sample%20images/"
)

sample_images = [
    "mdb215.png",
    "mdb216.png",
    "mdb217.png",
    "mdb218.png",
    "mdb219.png",
    "mdb220.png",
    "mdb221.png",
    "mdb222.png",
    "mdb223.png",
    "mdb224.png",
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
    "mdb224.png": "mdb224.png – Dense breast, Normal",
}

selected_filename = st.selectbox(
    "Select a sample mammogram for analysis",
    options=[""] + sample_images,
    format_func=lambda x: sample_labels.get(x, x) if x else "— Choose a sample —",
)


# ============================================================
# LOAD SELECTED SAMPLE
# ============================================================

selected_image = None

if selected_filename:
    image_url = GITHUB_RAW_BASE + selected_filename

    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        selected_image = Image.open(BytesIO(response.content)).convert("RGB")

        st.image(
            selected_image,
            caption=sample_labels[selected_filename],
            width="stretch",
        )
    except requests.RequestException:
        st.error(
            "❌ Failed to download the sample image. "
            "Please check the GitHub repository path."
        )
    except Exception:
        st.error("❌ The selected image could not be processed.")


# ============================================================
# USER UPLOAD
# ============================================================

st.markdown("### 📤 Or Upload Your Own Mammogram")

uploaded_file = st.file_uploader(
    "Upload digital mammogram (JPG/PNG/JPEG)",
    type=["jpg", "png", "jpeg"],
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file or selected_image:
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            source = "Uploaded Mammogram"
        except Exception:
            st.error("❌ The uploaded image could not be opened.")
            st.stop()
    else:
        image = selected_image
        source = "Selected Sample"

    col1, col2 = st.columns([1, 1.2])

    # ------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------
    with col1:
        st.image(image, caption=source, width="stretch")

    # ------------------------------------------------------
    # AI ANALYSIS
    # ------------------------------------------------------
    with col2:
        st.markdown("### 🔍 AI Analysis Result")

        with st.spinner("Analyzing mammographic features..."):
            try:
                prob = process_image(image)
            except Exception:
                st.error("❌ The AI model could not process this image.")
                st.stop()

        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
        st.markdown(f"**Computed Model Output: {prob:.1%}**")

        if prob >= 0.7:
            st.error("**HIGH MODEL-OUTPUT CATEGORY**")
            st.markdown(
                """
                **Interpretation:**

                The model produced a high probability score relative to the
                threshold configured in this application. This result should
                **not be interpreted as a cancer diagnosis**.

                **Next step:**

                A qualified healthcare professional should review the mammogram
                and determine whether additional diagnostic evaluation is
                appropriate.
                """
            )
        elif prob >= 0.5:
            st.warning("**INTERMEDIATE MODEL-OUTPUT CATEGORY**")
            st.markdown(
                """
                **Interpretation:**

                The model produced an intermediate probability score relative
                to the threshold configured in this application. The result
                does not establish whether a lesion is benign or malignant.

                **Next step:**

                Clinical correlation and appropriate professional interpretation
                are required.
                """
            )
        else:
            st.success("**LOWER MODEL-OUTPUT CATEGORY**")
            st.markdown(
                """
                **Interpretation:**

                The model produced a lower probability score relative to the
                threshold configured in this application. A lower model score
                does not rule out disease and should not be used as a diagnosis.

                **Next step:**

                Mammograms should be interpreted according to appropriate
                screening and clinical protocols.
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# BREAST CANCER INFORMATION
# ============================================================

st.markdown("## 🎗️ Breast Cancer Risk Factors & Prevention")

col_inf1, col_inf2 = st.columns(2)

with col_inf1:
    try:
        st.image(
            "https://www.iarc.who.int/wp-content/uploads/2023/10/BCAM_2_zoom.jpg",
            caption="Global Breast Cancer Burden",
            width="stretch",
        )
    except Exception:
        st.info("WHO/IARC information image could not be loaded.")

with col_inf2:
    try:
        st.image(
            "https://www.shutterstock.com/image-vector/breast-cancer-awareness-infographic-empowering-600nw-2355615993.jpg",
            caption="Breast Cancer Awareness",
            width="stretch",
        )
    except Exception:
        st.info("Awareness infographic could not be loaded.")


# ============================================================
# PREVENTION INFORMATION
# ============================================================

st.markdown("### Evidence-Based Prevention Strategies")

st.write(
    """
    - Follow breast cancer screening recommendations appropriate to age and
      individual risk.
    - Discuss personal and family history with a qualified healthcare
      professional.
    - Maintain a healthy lifestyle with balanced nutrition and regular
      physical activity.
    - Avoid tobacco exposure.
    - Limit alcohol consumption.
    - Discuss genetic risk assessment when there is a strong family history
      or other relevant risk factors.
    - Seek professional evaluation for new or concerning breast changes.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>System Developed by Simon</strong><br>
        Contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a>
        <br><br>
        © 2026 Breast Cancer AI Screening Assistant<br>
        Educational &amp; Research Platform • Global Health Awareness
    </div>
    """,
    unsafe_allow_html=True,
)
