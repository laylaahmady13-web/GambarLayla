
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi Tema
# ==========================
st.set_page_config(
    page_title="SmartVision - Image AI Dashboard",
    page_icon="🐾",
    layout="wide"
)

# Tema warna
bg_color = "#ffe6f2"
accent_color = "#b30086"
highlight_color = "#ff66b3"

# CSS custom
st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: #4b004b; }}
    .stSidebar {{ background-color: {bg_color}; }}
    h1 {{ color: {accent_color} !important; text-align:center; font-size: 28px !important; }}
    h2, h3, h4 {{ color: {accent_color} !important; text-align:center; }}
    div[data-testid="stFileUploaderDropzone"] {{
        background-color: #fff0f8 !important;
        border: 2px dashed {highlight_color} !important;
        border-radius: 12px;
    }}
    .result-card {{
        background: #fff0f8;
        padding: 20px;
        border-radius: 14px;
        border: 2px solid #ff99c8;
        box-shadow: 0px 4px 12px rgba(255, 182, 193, 0.5);
        text-align: center;
        margin-top: 20px;
        transition: all 0.3s ease;
    }}
    .result-card:hover {{
        transform: scale(1.04);
        box-shadow: 0px 6px 18px rgba(255, 102, 179, 0.6);
    }}
    .feedback-box {{
        background-color: #fff0f8;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #ff99c8;
        margin-top: 20px;
        text-align: center;
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
    ["Home", "Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Feedback Pengguna"]
)

# ==========================
# HOME
# ==========================
if menu == "Home":
    st.markdown("""
    <h2 style='text-align:center; color:#b30086;'>SmartVision Dashboard</h2>
    <p style='text-align:center; font-size:16px;'>
        Aplikasi AI sederhana yang memadukan dua model cerdas:
        <b>YOLO</b> untuk <i>deteksi objek</i> dan <b>CNN</b> untuk <i>klasifikasi gambar</i>.
        Model ini membantu mengenali perbedaan antara dua hewan yang sering tertukar — anjing dan serigala.
    </p>
    """, unsafe_allow_html=True)

    st.write("")  # spasi kecil

    # --- Dua gambar contoh kelas
    st.markdown("<h4 style='text-align:center;'>Contoh Dua Kelas yang Dikenali</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://cdn.pixabay.com/photo/2015/03/26/09/54/dog-690176_1280.jpg", 
                 caption="Kelas: Dog", use_container_width=True)
    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/09/07/08/57/wolf-2722407_1280.jpg", 
                 caption="Kelas: Wolf", use_container_width=True)

    st.markdown("""
    <hr>
    <h4 style='color:#b30086;'>🤖 Cara Kerja SmartVision</h4>
    <p style='font-size:15px; text-align:justify;'>
    <b>YOLO (You Only Look Once)</b> adalah model deteksi objek yang mampu menemukan letak objek di dalam gambar hanya dengan sekali pemindaian.
    YOLO menggambar kotak di sekitar objek yang dikenali, misalnya kepala atau tubuh hewan.
    </p>
    <p style='font-size:15px; text-align:justify;'>
    <b>CNN (Convolutional Neural Network)</b> digunakan untuk mengklasifikasikan jenis objek dalam gambar.
    CNN menganalisis fitur visual seperti bentuk telinga, warna bulu, dan struktur wajah untuk membedakan apakah gambar tersebut adalah
    <b>Dog</b> atau <b>Wolf</b>.
    </p>
    <hr>
    <p style='text-align:center; color:#b30086; font-size:15px;'>
        💡 Coba unggah gambar di menu “Deteksi Objek” atau “Klasifikasi Gambar” di sidebar kiri!
    </p>
    """, unsafe_allow_html=True)

# ==========================
# YOLO Detection
# ==========================
elif menu == "Deteksi Objek (YOLO)":
    st.markdown("""
<h3 style='color:#b30086;'>Deteksi Objek dengan YOLO</h3>
<p style='font-size:15px; text-align:justify;'>
YOLO mendeteksi posisi objek di dalam gambar hanya dengan satu kali pemrosesan. 
Coba unggah gambar hewan untuk melihat kotak deteksinya!
</p>
""", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Unggah gambar (bisa lebih dari satu):", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True)
            with st.spinner("Model YOLO sedang menganalisis..."):
                result_img = processor.predict_yolo(img)
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# ==========================
# CNN Classification
# ==========================
elif menu == "Klasifikasi Gambar":
    st.markdown("""
<h3 style='color:#b30086;'>Klasifikasi Gambar dengan CNN</h3>
<p style='font-size:15px; text-align:justify;'>
CNN mengenali pola visual seperti bentuk wajah dan warna bulu untuk memutuskan apakah hewan pada gambar adalah <b>Dog</b> atau <b>Wolf</b>.
</p>
""", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Unggah gambar (bisa lebih dari satu):", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        st.info(f"{len(uploaded_files)} gambar diunggah, sedang diproses...")
        emoji_map = {"Dog": "🐶", "Wolf": "🐺"}
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True)
            with st.spinner("Model CNN sedang memproses..."):
                label, conf = processor.predict_cnn(img)
                st.markdown(f"""
                    <div class="result-card">
                        <h3>{emoji_map.get(label, label)} {label}</h3>
                        <p style="font-size:18px; color:{accent_color};">
                            Confidence: {conf:.2f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)

# ==========================
# Feedback Pengguna
# ==========================
elif menu == "Feedback Pengguna":
    st.header("Bagikan Pendapatmu!")
    st.write("Kami ingin tahu seberapa puas kamu dengan dashboard ini.")
    
    rating = st.slider("Beri rating (1 = Buruk, 5 = Sempurna)", 1, 5, 3)
    feedback_text = st.text_area("Ketikkan komentar atau saranmu di sini:")
    
    if st.button("Kirim Feedback 🙌🏻"):
        st.success("Terima kasih atas feedback-nya!")
        st.balloons()
        st.session_state["feedback"] = {"rating": rating, "text": feedback_text}

    if "feedback" in st.session_state:
        fb = st.session_state["feedback"]
        st.markdown(f"""
            <div class="feedback-box">
                <h4>⭐ Rating kamu: {fb['rating']}/5</h4>
                <p><i>"{fb['text']}"</i></p>
            </div>
        """, unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b30086;'>🌸 Dashboard by Layla Ahmady Hsb | 2025 🌸</p>", unsafe_allow_html=True)
