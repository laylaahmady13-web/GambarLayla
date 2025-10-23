import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import plotly.express as px
from transformers import pipeline  # Untuk deskripsi gambar
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
    .stApp {{ background: linear-gradient(135deg, {bg_color}, #e6f7ff); color: #2e2e2e; font-family: 'Roboto', sans-serif; min-height: 100vh;}}
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
        self.caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

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
        caption = self.caption_model(img)[0]['generated_text']
        return caption

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
# ==========================
# HOME (SINTAKS YANG SUDAH DIPERBAIKI)
# ==========================
if menu == "Home":
    st.markdown("<h1>🚀 SmartVision Pro Dashboard</h1>", unsafe_allow_html=True)
    
    # 1. Pastikan Path Gambar Lokal Benar
    dog_img_path = "sample_images/n02102040_735_jpg.rf.c81ef292152e8e029218609b4a5fd235.jpg"
    wolf_img_path = "sample_images/animal-world-4069094__480_jpg.rf.c16604c33bd27dfedcf0a714aa8e140c.jpg"

    st.markdown("""
    <div class="home-section">
        <p style='text-align:center; font-size:20px; font-weight:bold; color:#2e2e2e;'>
            AI Dashboard Canggih untuk Deteksi & Klasifikasi Hewan: Anjing vs Serigala
        </p>
        <p style='text-align:center; font-size:16px; color:#666;'>
            Fitur Unik: YOLO untuk deteksi objek, CNN untuk klasifikasi, Deskripsi AI, dan Visualisasi!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center;'>Contoh Kelas dengan Deskripsi AI</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    # --- KOLOM DOG ---
    with col1:
        st.markdown('<div class="emoji-float">🐶</div>', unsafe_allow_html=True)
        # 2. st.image() menggunakan path lokal
        st.image(
            dog_img_path, 
            caption="**Dog** - Anjing", 
            use_container_width=True
        )
        if st.button("Deskripsikan Gambar 🐶"):
            with st.spinner("Generating caption..."):
                # 3. Image.open() menggunakan path lokal (INI YANG GAGAL SEBELUMNYA)
                img_to_caption = Image.open(dog_img_path)
                caption = processor.generate_caption(img_to_caption)
                st.write(f"**Deskripsi AI:** {caption}")
    
    # --- KOLOM WOLF ---
    with col2:
        st.markdown('<div class="emoji-float">🐺</div>', unsafe_allow_html=True)
        # 2. st.image() menggunakan path lokal
        st.image(
            wolf_img_path, 
            caption="**Wolf** - Serigala", 
            use_container_width=True
        )
        if st.button("Deskripsikan Gambar 🐺"):
            with st.spinner("Generating caption..."):
                # 3. Image.open() menggunakan path lokal (INI YANG GAGAL SEBELUMNYA)
                img_to_caption = Image.open(wolf_img_path)
                caption = processor.generate_caption(img_to_caption)
                st.write(f"**Deskripsi AI:** {caption}")

    with st.expander("🔍 Cara Kerja SmartVision Pro"):
        st.markdown("""
        - **YOLO**: Deteksi objek cepat dengan bounding box.
        - **CNN**: Klasifikasi berdasarkan fitur visual.
        - **AI Deskripsi**: Gunakan model BLIP untuk deskripsi gambar.
        - **Visualisasi**: Grafik distribusi hasil deteksi.
        """)

    st.markdown("<p style='text-align:center; color:#1e90ff; font-size:16px;'>Eksplorasi fitur unik di sidebar!</p>", unsafe_allow_html=True)
    
# ==========================
# YOLO Detection
# ==========================
elif menu == "Deteksi Objek (YOLO)":
    st.markdown("<h2>🔍 Deteksi Objek Canggih dengan YOLO</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:16px; color:#666;'>
        Unggah gambar untuk deteksi objek. Gunakan filter untuk fokus spesifik.
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True, caption="Gambar Asli")
            with st.spinner("🔄 Menganalisis dengan YOLO..."):
                result_img, boxes = processor.predict_yolo(img)
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
                
                filtered_boxes = [box for box in boxes if filter_option == "Semua" or processor.class_labels[int(box[5])] == filter_option]
                st.write(f"**Objek Terdeteksi (Filtered):** {len(filtered_boxes)}")
                
                if st.button("Export Hasil sebagai Gambar"):
                    result_img_pil = Image.fromarray(result_img)
                    result_img_pil.save("hasil_deteksi.png")
                    st.download_button("Download", data=open("hasil_deteksi.png", "rb"), file_name="hasil_deteksi.png")

# ==========================
# CNN Classification
# ==========================
elif menu == "Klasifikasi Gambar":
    st.markdown("<h2>🧠 Klasifikasi Gambar Canggih dengan CNN</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:16px; color:#666;'>
        Analisis pola visual + deskripsi AI untuk klasifikasi Dog atau Wolf.
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} gambar diproses...")
        emoji_map = {"Dog": "🐶", "Wolf": "🐺"}
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, use_container_width=True, caption="Gambar Asli")
            with st.spinner("🔄 Memproses dengan CNN + AI..."):
                label, conf = processor.predict_cnn(img)
                caption = processor.generate_caption(img)
                st.markdown(f"""
                    <div class="result-card">
                        <h3>{emoji_map.get(label, label)} {label}</h3>
                        <p style="font-size:20px; color:{accent_color}; font-weight:bold;">
                            Akurasi: {conf:.2f}%
                        </p>
                        <p style="font-size:16px; color:#666;">
                            <b>Deskripsi AI:</b> {caption}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

# ==========================
# Riwayat & Visualisasi
# ==========================
elif menu == "Riwayat & Visualisasi":
    st.markdown("<h2>📊 Riwayat & Visualisasi Hasil</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:16px; color:#666;'>
        Lihat distribusi hasil deteksi dan klasifikasi sebelumnya.
    </p>
    """, unsafe_allow_html=True)
    
    if "history" not in st.session_state:
        st.session_state.history = {"Dog": 5, "Wolf": 3}
    
    fig = px.pie(values=list(st.session_state.history.values()), names=list(st.session_state.history.keys()), title="Distribusi Klasifikasi")
    st.plotly_chart(fig)
    
    st.table(st.session_state.history)

# ==========================
# Feedback Pengguna
# ==========================
elif menu == "Feedback Pengguna":
    st.markdown("<h2>💬 Bagikan Pendapatmu!</h2>", unsafe_allow_html=True)
    st.write("Bantu kami tingkatkan dashboard unik ini.")
    
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
