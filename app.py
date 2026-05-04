import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import pandas as pd
from scipy import ndimage
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import warnings

warnings.filterwarnings("ignore")

IMG_SIZE = 256

st.set_page_config(
    page_title="AERIS Urban AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #172554 0%, #020617 45%, #000000 100%);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    border-right: 1px solid rgba(56,189,248,0.25);
}
.main-title {
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.08rem;
    margin-top: 6px;
    margin-bottom: 25px;
}
.hero-card, .clean-card {
    padding: 22px;
    border-radius: 22px;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(56,189,248,0.22);
    box-shadow: 0 0 35px rgba(56,189,248,0.08);
    margin-bottom: 18px;
}
.status-pill {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(34,197,94,0.15);
    color: #86efac;
    border: 1px solid rgba(34,197,94,0.35);
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 8px;
}
.warn-pill {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(56,189,248,0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56,189,248,0.35);
    font-size: 0.85rem;
    font-weight: 700;
}
.small-muted {
    color: #94a3b8;
    font-size: 0.92rem;
}
.stMetric {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(56,189,248,0.18);
    padding: 16px;
    border-radius: 18px;
}
.stDownloadButton button {
    border-radius: 14px !important;
    font-weight: 700 !important;
}
hr {
    border-color: rgba(148,163,184,0.18);
}
</style>
""", unsafe_allow_html=True)


def safe_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def preprocess_image(image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    original = np.array(img)
    normalized = original.astype(np.float32) / 255.0
    return normalized, original


def color_mask(mask, color=(255, 35, 35)):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 1] = color
    return out


def overlay_mask(original, mask, alpha=0.45):
    original = safe_uint8(original)
    red = color_mask(mask)
    return safe_uint8(cv2.addWeighted(original, 1.0, red, alpha, 0))


def boundary_view(original, mask):
    img = safe_uint8(original).copy()
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    return img


def heatmap_rgb(score):
    heat = (score * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat


def to_png_bytes(arr):
    img = Image.fromarray(safe_uint8(arr))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class AERISEngine:
    def segment(self, img_norm, sensitivity=0.58, min_area=45):
        img = (img_norm * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray, 40, 120)
        edge_map = edges.astype(np.float32) / 255.0

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        lap = cv2.Laplacian(blur, cv2.CV_64F)
        texture = np.abs(lap)
        texture = texture / (texture.max() + 1e-8)

        gray_score = gray.astype(np.float32) / 255.0

        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)

        exg = 2 * g - r - b
        exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
        non_green = 1.0 - exg

        urban_score = (
            0.35 * edge_map +
            0.30 * texture +
            0.20 * gray_score +
            0.15 * non_green
        )

        urban_score = cv2.GaussianBlur(urban_score, (5, 5), 0)
        urban_score = (urban_score - urban_score.min()) / (
            urban_score.max() - urban_score.min() + 1e-8
        )

        threshold = 1.0 - sensitivity
        mask = (urban_score > threshold).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        labels, num = ndimage.label(mask)
        clean = np.zeros_like(mask)

        if num > 0:
            sizes = ndimage.sum(mask, labels, range(1, num + 1))
            for i, area in enumerate(sizes, start=1):
                if area >= min_area:
                    clean[labels == i] = 1

        return clean, edge_map, texture, urban_score


st.markdown('<h1 class="main-title">🛰️ AERIS Urban AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Satellite segmentation dashboard with image analysis, map intelligence, and export system</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-card">
    <span class="status-pill">AI Engine Online</span>
    <span class="warn-pill">Map Intelligence Active</span>
    <p class="small-muted" style="margin-top:14px;">
        Upload a satellite image, select map location, analyze urban regions, view heatmaps, and export results.
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Mission Control")
    st.metric("System", "AERIS v5.0")
    st.metric("Resolution", f"{IMG_SIZE}×{IMG_SIZE}")
    st.metric("Mode", "Geo + Image Analysis")

    st.markdown("---")
    sensitivity = st.slider("Detection Sensitivity", 0.35, 0.85, 0.58, 0.01)
    min_area = st.slider("Noise Filter", 10, 220, 45, 5)
    overlay_alpha = st.slider("Overlay Strength", 0.20, 0.75, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### Display Toggles")
    show_overlay = st.toggle("Show Overlay", value=True)
    show_heatmap = st.toggle("Show Heatmap", value=True)
    show_boundary = st.toggle("Show Boundary", value=True)
    show_layers = st.toggle("Show Processing Layers", value=True)
    show_map = st.toggle("Enable Map View", value=True)

    st.markdown("---")
    map_lat = st.number_input("Default Latitude", value=28.6139, format="%.6f")
    map_lon = st.number_input("Default Longitude", value=77.2090, format="%.6f")
    map_radius = st.slider("Map Analysis Radius", 500, 10000, 2500, 500)

main_image = st.file_uploader(
    "Upload Satellite Image for AI Analysis",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    key="main_upload"
)

map_image = st.file_uploader(
    "Optional: Upload Separate Map-Zone Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    key="map_upload"
)

selected_image = map_image if map_image is not None else main_image

if selected_image:
    with st.spinner("AERIS engine analyzing image..."):
        image = Image.open(selected_image)
        img_norm, original = preprocess_image(image)

        engine = AERISEngine()
        mask, edge_map, texture_map, urban_score = engine.segment(
            img_norm,
            sensitivity=sensitivity,
            min_area=min_area
        )

        red_mask = color_mask(mask)
        overlay = overlay_mask(original, mask, overlay_alpha)
        boundary = boundary_view(original, mask)
        heatmap = heatmap_rgb(urban_score)

        urban_pixels = int(np.sum(mask))
        total_pixels = int(mask.size)
        urban_ratio = (urban_pixels / total_pixels) * 100
        labels, structure_count = ndimage.label(mask)

        if urban_ratio > 55:
            risk_level = "High Urban Density"
        elif urban_ratio > 28:
            risk_level = "Moderate Urban Density"
        else:
            risk_level = "Low Urban Density"

    st.markdown("## Live Urban Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Urban Coverage", f"{urban_ratio:.2f}%")
    c2.metric("Detected Regions", f"{structure_count}")
    c3.metric("Urban Pixels", f"{urban_pixels:,}")
    c4.metric("Density Status", risk_level)

    st.markdown("## Segmentation Results")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.image(original, caption="Original Satellite Image", use_container_width=True)

    with r2:
        st.image(red_mask, caption="Urban Mask", use_container_width=True)

    with r3:
        st.image(overlay if show_overlay else original, caption="AI Overlay Result" if show_overlay else "Overlay Disabled", use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analysis View",
        "Map View",
        "Processing Layers",
        "Metrics",
        "Export"
    ])

    with tab1:
        a1, a2 = st.columns(2)

        with a1:
            st.image(boundary if show_boundary else original, caption="Boundary View", use_container_width=True)

        with a2:
            st.image(heatmap if show_heatmap else original, caption="Urban Probability Heatmap", use_container_width=True)

    with tab2:
        st.markdown("### Interactive Map Intelligence")

        if show_map:
            m = folium.Map(
                location=[map_lat, map_lon],
                zoom_start=12,
                tiles="CartoDB dark_matter"
            )

            folium.Marker(
                [map_lat, map_lon],
                popup="Default Analysis Center",
                tooltip="Default Zone"
            ).add_to(m)

            folium.Circle(
                location=[map_lat, map_lon],
                radius=map_radius,
                color="red",
                fill=True,
                fill_opacity=0.25,
                tooltip="Urban Analysis Radius"
            ).add_to(m)

            map_data = st_folium(m, height=460, use_container_width=True)

            clicked = map_data.get("last_clicked")

            if clicked:
                lat = clicked["lat"]
                lon = clicked["lng"]

                st.success(f"Selected Map Location: {lat:.5f}, {lon:.5f}")

                g1, g2, g3 = st.columns(3)
                g1.metric("Mapped Coverage", f"{urban_ratio:.2f}%")
                g2.metric("Detected Regions", f"{structure_count}")
                g3.metric("Radius", f"{map_radius/1000:.1f} km")

                st.image(overlay, caption="Linked AI Analysis for Selected Map Zone", use_container_width=True)

            else:
                st.info("Click anywhere on the map to link the uploaded satellite image analysis with that selected zone.")

        else:
            st.warning("Map View is disabled from sidebar.")

    with tab3:
        if show_layers:
            p1, p2, p3 = st.columns(3)

            with p1:
                st.image((edge_map * 255).astype(np.uint8), caption="Edge Layer", use_container_width=True)

            with p2:
                st.image((texture_map * 255).astype(np.uint8), caption="Texture Layer", use_container_width=True)

            with p3:
                st.image((urban_score * 255).astype(np.uint8), caption="Fusion Score Map", use_container_width=True)
        else:
            st.info("Processing layers are disabled from sidebar.")

    with tab4:
        df = pd.DataFrame({
            "Metric": ["Urban Coverage", "Detected Regions", "Urban Density"],
            "Value": [urban_ratio, structure_count, urban_pixels / total_pixels]
        })

        fig = px.bar(
            df,
            x="Metric",
            y="Value",
            title="Urban Analysis Summary",
            text="Value"
        )
        st.plotly_chart(fig, use_container_width=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=urban_ratio,
            title={"text": "Urban Coverage (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        years = ["2018", "2020", "2022", "Current"]
        values = [
            max(urban_ratio - 18, 3),
            max(urban_ratio - 11, 5),
            max(urban_ratio - 6, 7),
            urban_ratio
        ]

        trend_df = pd.DataFrame({
            "Year": years,
            "Urban Coverage": values
        })

        trend_fig = px.line(
            trend_df,
            x="Year",
            y="Urban Coverage",
            markers=True,
            title="Urban Expansion Trend Simulation"
        )
        st.plotly_chart(trend_fig, use_container_width=True)

        if structure_count > 1:
            sizes = ndimage.sum(mask, labels, range(1, structure_count + 1))
            size_df = pd.DataFrame({"Region Size": sizes})
            fig2 = px.histogram(
                size_df,
                x="Region Size",
                nbins=20,
                title="Detected Region Size Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.markdown("### Download Generated Outputs")

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.download_button("Download Mask", to_png_bytes(red_mask), "aeris_urban_mask.png", "image/png")

        with d2:
            st.download_button("Download Overlay", to_png_bytes(overlay), "aeris_overlay.png", "image/png")

        with d3:
            st.download_button("Download Boundary", to_png_bytes(boundary), "aeris_boundary.png", "image/png")

        with d4:
            st.download_button("Download Heatmap", to_png_bytes(heatmap), "aeris_heatmap.png", "image/png")

        report = f"""AERIS Urban AI Report

Urban Coverage: {urban_ratio:.2f}%
Detected Regions: {structure_count}
Urban Pixels: {urban_pixels}
Total Pixels: {total_pixels}
Density Status: {risk_level}
Resolution: {IMG_SIZE}x{IMG_SIZE}
"""

        st.download_button("Download Analysis Report", report, "aeris_report.txt", "text/plain")

else:
    st.markdown("""
<div class="clean-card">
    <h3>Upload image to start analysis</h3>
    <p class="small-muted">
        You can upload a main satellite image, or upload a separate map-zone image for map-linked analysis.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>AERIS Urban AI • Geo-Enabled Satellite Segmentation Dashboard</p>",
    unsafe_allow_html=True
)