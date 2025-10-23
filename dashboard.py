import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import random

# ==========================
# Tema Gradient Full & Random
# ==========================
st.set_page_config(
    page_title="SmartVision Unique Dashboard",
    page_icon="🚀",
    layout="wide"
)

gradient_colors = [
    ["#ffe6f2", "#ffc6e0"],
    ["#e0f7fa", "#80deea"],
    ["#fff3e0", "#ffcc80"],
    ["#f3e5f5", "#ce93d8"]
]
bg = random.choice(gradient_colors)

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(135deg, {bg[0]}, {bg[1]});
    color: #2e2e2e;
    font-family: 'Roboto', sans-serif;
}}
.stSidebar {{
    background: linear-gradient(135deg, {bg[0]}, {bg[1]});
    border-right: 4px solid #ff69b4;
}}
h1 {{
    color: #ff1493 !important;
    text-align:center; 
    font-size: 36px !important;
    font-weight: bold; 
    animation: pulse 2s infinite;
}}
.result-card {{
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 3px solid #ff69b4;
    box-shadow: 0px 4px 15px rgba(255,105,180,0.3);
    text-align: center;
    margin-top: 20px;
    transition: all 0.3s ease-in-out;
}}
.result-card:hover {{
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0px 8px 25px rgba(255,105,180,0.5);
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
    from {{ text-shadow: 0 0 5px #ff69b4; }}
    to {{ text-shadow: 0 0 20px #ff1493, 0 0 30px #ff69b4; }}
}}
@keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
    100% {{ transform: scale(1); }}
}}
</style>
""", unsafe_allow_html=True)

# ==========================
# Image Processor Class
# ==========================
class ImageProcessor:
    def __init__(self, yolo_model, cnn_model, class_labels):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.class_labels = class_labels

    def predict_yolo(self, img: Image.Image):
        results = self.yolo_model(img)
        return results[0].plot()

    def predict_cnn(self, img: Image.Image):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        prediction = self.cnn_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return self.class_labels[class_index], confidence

# ==========================
# Load Models
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
menu = st.sidebar.radio("Pilih Mode:", ["Home", "Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# ==========================
# HOME
# ==========================
if menu == "Home":
    st.markdown("<h1>🚀 SmartVision Unique Dashboard</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="emoji-float">🐶</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="emoji-float">🐺</div>', unsafe_allow_html=True)
    st.markdown("Aplikasi AI untuk **Deteksi Objek** dan **Klasifikasi Gambar**: Anjing vs Serigala")

# ==========================
# YOLO Detection
# ==========================
elif menu == "Deteksi Objek (YOLO)":
    st.header("Deteksi Objek dengan YOLO")
    uploaded_files = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_column_width=True, caption="Gambar Asli")
            with st.spinner("Mendeteksi dengan YOLO..."):
                result_img = processor.predict_yolo(img)
                st.image(result_img, use_column_width=True, caption="Hasil Deteksi")

# ==========================
# CNN Classification
# ==========================
elif menu == "Klasifikasi Gambar":
    st.header("Klasifikasi Gambar dengan CNN")
    uploaded_files = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_column_width=True, caption="Gambar Asli")
            with st.spinner("Memproses CNN..."):
                label, conf = processor.predict_cnn(img)
                emoji_map = {"Dog":"🐶", "Wolf":"🐺"}
                st.markdown(f"""
                    <div class="result-card">
                        <h3>{emoji_map.get(label,label)} {label}</h3>
                        <p style="font-size:20px; color:#ff1493;"><b>Akurasi:</b> {conf:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ff1493;'>Dashboard by Layla Ahmady Hsb 🔥</p>", unsafe_allow_html=True)
