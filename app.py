import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

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
st.markdown("### AI Powered Unsupervised Segmentation Engine (VAE Neural Mapping)")
st.markdown("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🚀 Mission Control")
st.sidebar.metric("System", "AERIS v2.0")
st.sidebar.metric("AI Mode", "Unsupervised VAE")
st.sidebar.metric("Input", "128x128 RGB")
st.sidebar.success("Status: ONLINE 🟢")

# =========================
# MODEL
# =========================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_encoder():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = tf.keras.layers.Conv2D(64, 3, 2, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    z_mean = tf.keras.layers.Dense(LATENT_DIM)(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM)(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model(inp, [z_mean, z_log_var, z])


@st.cache_resource
def load_models():
    encoder = build_encoder()

    encoder.load_weights("saved_models/advanced_urban_encoder.h5")

    return encoder


encoder = load_models()

# =========================
# PROCESSING (NO cv2 FIXED)
# =========================
def preprocess(img):
    img = Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)


def generate_mask(img):
    z_mean, z_log_var, z = encoder.predict(img, verbose=0)

    latent = np.abs(z[0])
    size = int(np.sqrt(len(latent)))
    size = max(4, size)

    latent_map = latent[:size * size].reshape(size, size)

    latent_map = np.array(
        Image.fromarray(latent_map).resize((IMG_SIZE, IMG_SIZE))
    )

    latent_map = (latent_map - latent_map.min()) / (latent_map.max() + 1e-7)

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
            st.image(image, caption="Original Image")

        with col2:
            st.image(mask * 255, caption="AI Segmentation Mask")

    with tab2:
        fig, ax = plt.subplots()
        ax.imshow(latent_map, cmap="jet")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.hist(latent_map.flatten(), bins=30)
        st.pyplot(fig2)

    with tab3:
        st.info("VAE Neural System Active")
        st.success("Status: STABLE 🚀")

st.markdown("---")
st.markdown("AERIS NASA AI • VAE Neural Engine")