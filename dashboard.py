import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import plotly.express as px
import time

# ==========================
# Konfigurasi Tema Unik
# ==========================
st.set_page_config(
    page_title="SmartVision Pro - AI Dashboard Unik",
    page_icon="🚀",
    layout="wide"
)

bg_color = "#f0f8ff"
accent_color = "#1e90ff"
highlight_color = "#00bfff"
glow_color = "#87ceeb"

st.markdown(f"""
    <style>
    .stApp {{ background: linear-gradient(135deg, {bg_color}, #e6f7ff); color: #2e2e2e; font-family: 'Roboto', sans-serif; min-height: 100vh; }}
    .stSidebar {{ background: linear-gradient(135deg, {bg_color}, #d1ecf1); border-right: 4px solid {highlight_color}; box-shadow: 0 0 20px {glow_color}; }}
    h1 {{ color: {accent_color} !important; text-align:center; font-size: 36px !important; font-weight: bold; text-shadow: 0 0 10px {glow_color}; animation: pulse 2s infinite; }}
    h2, h3, h4 {{ color: {accent_color} !important; text-align:center; font-weight: bold; }}
    div[data-testid="stFileUploaderDropzone"] {{
        background: linear-gradient(135deg, #ffffff, #f0f8ff) !important;
        border: 3px dashed {highlight_color} !important;
        border-radius: 20px;
        transition: all 0.4s ease;
        box-shadow: 0 0 15px {glow_color};
    }}
    div[data-testid="stFileUploaderDropzone"]:hover {{
        background: linear-gradient(135deg, #e6f7ff, #ffffff) !important;
        border-color: {accent_color} !important;
        transform: scale(1.05);
        box-shadow: 0 0 25px {accent_color};
    }}
    .result-card {{
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        padding: 30px;
        border-radius: 25px;
        border: 3px solid {highlight_color};
        box-shadow: 0 8px 20px rgba(30, 144, 255, 0.3);
        text-align: center;
        margin-top: 25px;
        transition: all 0.5s ease;
        position: relative;
        overflow: hidden;
    }}
    .result-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
        transition: left 0.6s;
    }}
    .result-card:hover::before {{
        left: 100%;
    }}
    .result-card:hover {{
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 12px 30px rgba(30, 144, 255, 0.5);
    }}
    .feedback-box {{
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        border-radius: 20px;
        padding: 25px;
        border: 2px solid {highlight_color};
        margin-top: 25px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }}
    .home-section {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        border: 1px solid {glow_color};
    }}
    .emoji-float {{
        font-size: 60px;
        animation: float 3s ease-in-out infinite, glow 2s infinite alternate;
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-15px); }}
    }}
    @keyframes glow {{
        from {{ text-shadow: 0 0 5px {glow_color}; }}
        to {{ text-shadow: 0 0 20px {accent_color}, 0 0 30px {accent_color}; }}
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================
# Class Pemroses Gambar
# ==========================
class ImageProcessor:
    def __init__(self, yolo_model, cnn_model, class_labels):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.class_labels = class_labels
        self.caption_model = None  # pipeline caption dinonaktifkan agar aman di cloud

    def predict_yolo(self, img: Image.Image):
        results = self.yolo_model(img)
        return results[0].plot(), results[0].boxes.data.tolist()

    def predict_cnn(self, img: Image.Image):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        prediction = self.cnn_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return self.class_labels[class_index], confidence

    def generate_caption(self, img: Image.Image):
        return "Deskripsi AI nonaktif pada versi ini."

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Layla Ahmady Hsb_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Layla Ahmady Hsb_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()
processor = ImageProcessor(yolo_model, classifier, ["Dog", "Wolf"])

# ==========================
# Sidebar Menu
# ==========================
menu = st.sidebar.radio(
    "Pilih Mode:",
    ["Home", "Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Riwayat & Visualisasi", "Feedback Pengguna"]
)

filter_option = st.sidebar.selectbox("Filter Deteksi (Opsional):", ["Semua", "Dog", "Wolf"])

# =================
