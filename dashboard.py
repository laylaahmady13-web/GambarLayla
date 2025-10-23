import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px

# ==========================
# Konfigurasi Tema
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
h1 {{ color: {accent_color} !important; text-align:center; font-size: 36px !important; font-weight: bold; text-shadow: 0 0 10px {glow_color}; }}
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
        self.caption_model = None  # pipeline caption dihapus sepenuhnya

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
        return "Deskripsi AI nonaktif."

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

# ==========================
# HOME
# ==========================
if menu == "Home":
    st.markdown("<h1>🚀 SmartVision Pro Dashboard</h1>", unsafe_allow_html=True)
    dog_img_path = "sample_images/dog.jpg"
    wolf_img_path = "sample_images/wolf.jpg"

    st.markdown("""
    <div class="home-section">
        <p style='text-align:center; font-size:20px; font-weight:bold; color:#2e2e2e;'>
            AI Dashboard: Anjing vs Serigala
        </p>
        <p style='text-align:center; font-size:16px; color:#666;'>
            Fitur: YOLO deteksi objek & CNN klasifikasi
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(dog_img_path, caption="Dog - Anjing", use_container_width=True)
    with col2:
        st.image(wolf_img_path, caption="Wolf - Serigala", use_container_width=True)

# ==========================
# YOLO Detection
# ==========================
elif menu == "Deteksi Objek (YOLO)":
    st.markdown("<h2>Deteksi Objek dengan YOLO</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True, caption="Gambar Asli")
            with st.spinner("Menganalisis..."):
                result_img, boxes = processor.predict_yolo(img)
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
                filtered_boxes = [box for box in boxes if filter_option == "Semua" or processor.class_labels[int(box[5])] == filter_option]
                st.write(f"Objek Terdeteksi (Filtered): {len(filtered_boxes)}")

# ==========================
# CNN Classification
# ==========================
elif menu == "Klasifikasi Gambar":
    st.markdown("<h2>Klasifikasi Gambar dengan CNN</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        emoji_map = {"Dog": "🐶", "Wolf": "🐺"}
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True, caption="Gambar Asli")
            label, conf = processor.predict_cnn(img)
            caption = processor.generate_caption(img)
            st.markdown(f"""
            <div class="result-card">
                <h3>{emoji_map.get(label,label)} {label}</h3>
                <p>Akurasi: {conf:.2f}%</p>
                <p>{caption}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================
# Riwayat & Visualisasi
# ==========================
elif menu == "Riwayat & Visualisasi":
    st.markdown("<h2>Riwayat & Visualisasi</h2>", unsafe_allow_html=True)
    if "history" not in st.session_state:
        st.session_state.history = {"Dog": 5, "Wolf": 3}
    fig = px.pie(values=list(st.session_state.history.values()), names=list(st.session_state.history.keys()), title="Distribusi Klasifikasi")
    st.plotly_chart(fig)
    st.table(st.session_state.history)

# ==========================
# Feedback Pengguna
# ==========================
elif menu == "Feedback Pengguna":
    st.markdown("<h2>Feedback Pengguna</h2>", unsafe_allow_html=True)
    rating = st.slider("Rating (1-5)", 1, 5, 3)
    feedback_text = st.text_area("Komentar atau saran:")
    if st.button("Kirim Feedback"):
        st.success("Terima kasih!")
        st.session_state["feedback"] = {"rating": rating, "text": feedback_text}
    if "feedback" in st.session_state:
        fb = st.session_state["feedback"]
        st.markdown(f"<div class='feedback-box'><b>Rating:</b> {fb['rating']}<br><i>{fb['text']}</i></div>", unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Dashboard by Layla Ahmady Hsb | 2025 🐾</p>", unsafe_allow_html=True)
