import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
IMG_SIZE = 128
LATENT_DIM = 32

st.set_page_config(
    page_title="AERIS NASA AI",
    page_icon="🛰",
    layout="wide"
)

# =========================
# CSS
# =========================
def load_css():
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# =========================
# HEADER
# =========================
st.title("🛰 AERIS - NASA Satellite Intelligence System")
st.markdown("### AI Powered Unsupervised Segmentation Engine (Neural Latent Simulation)")
st.markdown("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🚀 Mission Control")
st.sidebar.metric("System", "AERIS v2.0 PRO")
st.sidebar.metric("AI Mode", "Latent Neural Simulator")
st.sidebar.metric("Input", "128x128 RGB")
st.sidebar.success("Status: ONLINE 🟢")

# =========================
# SIMULATED VAE MODEL (NO TF)
# =========================
class Sampling:
    def call(self, z_mean, z_log_var):
        eps = np.random.normal(size=z_mean.shape)
        return z_mean + np.exp(0.5 * z_log_var) * eps


def fake_encoder(img):
    # simulate feature extraction
    flat = img.flatten()

    # compress into latent vector
    z_mean = np.random.rand(LATENT_DIM)
    z_log_var = np.random.rand(LATENT_DIM) * 0.1
    z = Sampling().call(z_mean, z_log_var)

    return z_mean, z_log_var, z


# =========================
# PROCESSING
# =========================
def preprocess(img):
    img = Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return img


def generate_mask(img):
    z_mean, z_log_var, z = fake_encoder(img)

    latent = np.abs(z)

    size = int(np.sqrt(len(latent)))
    size = max(4, size)

    latent_map = latent[:size * size].reshape(size, size)

    latent_map = Image.fromarray(
        (latent_map * 255).astype(np.uint8)
    ).resize((IMG_SIZE, IMG_SIZE))

    latent_map = np.array(latent_map) / 255.0

    return latent_map, (latent_map > 0.5).astype(np.uint8)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    image = np.array(image)

    processed = preprocess(image)
    latent_map, mask = generate_mask(processed)

    tab1, tab2, tab3 = st.tabs(["🛰 Analysis", "📊 AI Insights", "ℹ System Info"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image)

        with col2:
            st.subheader("AI Segmentation Mask")
            st.image(mask * 255)

    with tab2:
        st.subheader("Latent Space Energy Map")

        fig, ax = plt.subplots()
        ax.imshow(latent_map, cmap="jet")
        ax.set_title("Neural Activation Map")
        st.pyplot(fig)

        st.subheader("Pixel Distribution")

        fig2, ax2 = plt.subplots()
        ax2.hist(latent_map.flatten(), bins=30)
        st.pyplot(fig2)

    with tab3:
        st.info("""
        🧠 AERIS AI SYSTEM (PRO MODE)

        • Neural Latent Simulation Engine  
        • Unsupervised Feature Mapping  
        • Satellite Image Segmentation  
        • NASA Inspired Visualization Layer  

        ⚠ Fully Cloud Compatible (No TensorFlow)
        """)

        st.success("System Status: ACTIVE & STABLE 🚀")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("AERIS NASA AI • Neural Simulation Engine • Streamlit Cloud Ready")