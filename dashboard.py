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

# CSS custom untuk tampilan lebih rapi dan unik
st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: #4b004b; font-family: 'Arial', sans-serif; }}
    .stSidebar {{ background-color: {bg_color}; border-right: 3px solid {highlight_color}; }}
    h1 {{ color: {accent_color} !important; text-align:center; font-size: 32px !important; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }}
    h2, h3, h4 {{ color: {accent_color} !important; text-align:center; font-weight: bold; }}
    div[data-testid="stFileUploaderDropzone"] {{
        background-color: #fff0f8 !important;
        border: 2px dashed {highlight_color} !important;
        border-radius: 15px;
        transition: all 0.3s ease;
    }}
    div[data-testid="stFileUploaderDropzone"]:hover {{
        background-color: #ffe6f2 !important;
        border-color: {accent_color} !important;
    }}
    .result-card {{
        background: linear-gradient(135deg, #fff0f8, #ffe6f2);
        padding: 25px;
        border-radius: 20px;
        border: 2px solid #ff99c8;
        box-shadow: 0px 6px 15px rgba(255, 182, 193, 0.4);
        text-align: center;
        margin-top: 20px;
        transition: all 0.4s ease;
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
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s;
    }}
    .result-card:hover::before {{
        left: 100%;
    }}
    .result-card:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0px 10px 25px rgba(255, 102, 179, 0.5);
    }}
    .feedback-box {{
        background-color: #fff0f8;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #ff99c8;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }}
    .home-section {{
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }}
    .emoji-float {{
        font-size: 50px;
        animation: float 3s ease-in-out infinite;
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
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
    st.markdown("<h1>🐾 SmartVision Dashboard</h1>", unsafe_allow_html=True)
    
    # Intro singkat
    st.markdown("""
    <div class="home-section">
        <p style='text-align:center; font-size:18px; font-weight:bold; color:#4b004b;'>
            AI Dashboard untuk Deteksi & Klasifikasi Hewan: Anjing vs Serigala
        </p>
        <p style='text-align:center; font-size:16px; color:#666;'>
            Gunakan YOLO untuk deteksi objek dan CNN untuk klasifikasi. Eksplorasi fitur di sidebar!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Gambar contoh dengan layout rapi
    st.markdown("<h3 style='text-align:center;'>Contoh Kelas</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="emoji-float">🐶</div>', unsafe_allow_html=True)
        st.image(
            "https://cdn.pixabay.com/photo/2017/08/01/09/04/dog-2563759_1280.jpg", 
            caption="**Dog** - Anjing", 
            use_container_width=True
        )
        
    with col2:
        st.markdown('<div class="emoji-float">🐺</div>', unsafe_allow_html=True)
        st.image(
            "https://cdn.pixabay.com/photo/2023/11/07/12/55/wolf-8372315_1280.jpg", 
            caption="**Wolf** - Serigala", 
            use_container_width=True
        )

    # Cara kerja singkat dengan expander untuk detail
    with st.expander("🔍 Cara Kerja SmartVision"):
        st.markdown("""
        - **YOLO**: Deteksi objek dengan kotak bounding box dalam satu pemindaian.
        - **CNN**: Klasifikasi berdasarkan fitur visual seperti bentuk dan warna.
        """)

    st.markdown("<p style='text-align:center; color:#b30086; font-size:16px;'>Mulai dengan mengunggah gambar di menu sidebar!</p>", unsafe_allow_html=True)

# ==========================
# YOLO Detection
# ==========================
elif menu == "Deteksi Objek (YOLO)":
    st.markdown("<h2>🔍 Deteksi Objek dengan YOLO</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:16px; color:#666;'>
        Unggah gambar hewan untuk melihat deteksi objek secara real-time.
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Unggah gambar (bisa lebih dari satu):", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True, caption="Gambar Asli")
            with st.spinner("🔄 Menganalisis dengan YOLO..."):
                result_img = processor.predict_yolo(img)
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# ==========================
# CNN Classification
# ==========================
elif menu == "Klasifikasi Gambar":
    st.markdown("<h2>🧠 Klasifikasi Gambar dengan CNN</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:16px; color:#666;'>
        Analisis pola visual untuk klasifikasi Dog atau Wolf.
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Unggah gambar (bisa lebih dari satu):", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} gambar diproses...")
        emoji_map = {"Dog": "🐶", "Wolf": "🐺"}
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True, caption="Gambar Asli")
            with st.spinner("🔄 Memproses dengan CNN..."):
                label, conf = processor.predict_cnn(img)
                st.markdown(f"""
                    <div class="result-card">
                        <h3>{emoji_map.get(label, label)} {label}</h3>
                        <p style="font-size:20px; color:{accent_color}; font-weight:bold;">
                            Akurasi: {conf:.2f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)

# ==========================
# Feedback Pengguna
# ==========================
elif menu == "Feedback Pengguna":
    st.markdown("<h2>Bagikan Pendapatmu!</h2>", unsafe_allow_html=True)
    st.write("Bantu kami tingkatkan dashboard ini.")
    
    rating = st.slider("⭐ Rating (1 = Buruk, 5 = Sempurna)", 1, 5, 3)
    feedback_text = st.text_area("Komentar atau saran:")
    
    if st.button("Kirim Feedback 🙌"):
        st.success("Terima kasih atas feedback-nya!")
        st.balloons()
        st.session_state["feedback"] = {"rating": rating, "text": feedback_text}

    if "feedback" in st.session_state:
        fb = st.session_state["feedback"]
        st.markdown(f"""
            <div class="feedback-box">
                <h4>⭐ Rating: {fb['rating']}/5</h4>
                <p><i>"{fb['text']}"</i></p>
            </div>
        """, unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b30086; font-size:14px;'>Dashboard by Layla Ahmady Hsb | 2025 🐾</p>", unsafe_allow_html=True)
