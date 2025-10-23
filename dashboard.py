import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

# ==========================
# Konfigurasi Tema Unik
# ==========================
st.set_page_config(
    page_title="SmartVision Unik",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background: linear-gradient(120deg, #f0f8ff, #e0f7ff);}
h1 {text-align:center; color:#1e90ff; font-size:36px;}
.result-card {background:#ffffffaa; padding:20px; border-radius:15px; margin-top:15px; text-align:center; box-shadow:0px 4px 15px #87ceeb;}
.result-card:hover {transform:scale(1.03); box-shadow:0px 6px 25px #1e90ff;}
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/model_yolo.pt")
    cnn_model = tf.keras.models.load_model("model/model_cnn.h5")
    return yolo_model, cnn_model

yolo_model, cnn_model = load_models()
class_labels = ["Dog", "Wolf"]


# Class untuk prediksi
class ImageProcessor:
    def __init__(self, yolo_model, cnn_model, class_labels):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.class_labels = class_labels

    def predict_yolo(self, img: Image.Image):
        results = self.yolo_model(img)
        return results[0].plot(), results[0].boxes.data.tolist()

    def predict_cnn(self, img: Image.Image):
        img_resized = img.resize((224,224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)/255.0
        pred = self.cnn_model.predict(img_array)
        idx = np.argmax(pred)
        return self.class_labels[idx], np.max(pred)*100

processor = ImageProcessor(yolo_model, cnn_model, class_labels)


# Sidebar Menu
menu = st.sidebar.radio("Pilih Mode:", ["Home","Deteksi YOLO","Klasifikasi CNN"])

# HOME
if menu=="Home":
    st.markdown("<h1>🚀 SmartVision Unik</h1>", unsafe_allow_html=True)
    st.write("Aplikasi deteksi objek dan klasifikasi gambar dengan YOLO & CNN.")

    st.markdown("""
    <div style="background:#ffffffaa; padding:15px; border-radius:15px; text-align:center; box-shadow:0px 4px 15px #87ceeb;">
        <b>YOLO:</b> Digunakan untuk mendeteksi objek di gambar (misal anjing 🐶 dan serigala 🐺).<br>
        <b>CNN:</b> Digunakan untuk mengklasifikasikan gambar menjadi kelas Dog atau Wolf.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.image("sample_images/dog.jpg", caption="Dog")
    col2.image("sample_images/wolf.jpg", caption="Wolf")


# YOLO Detection
elif menu=="Deteksi YOLO":
    st.header("Deteksi Objek Menggunakan YOLO")
    uploaded_files = st.file_uploader("Unggah gambar", type=["jpg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, caption=f.name)
            result_img, boxes = processor.predict_yolo(img)
            st.image(result_img, caption="Hasil Deteksi")
            st.write(f"Objek Terdeteksi: {len(boxes)}")


# CNN Classification
elif menu=="Klasifikasi CNN":
    st.header("Klasifikasi Gambar Menggunakan CNN")
    uploaded_files = st.file_uploader("Unggah gambar", type=["jpg","png"], accept_multiple_files=True)
    emoji_map = {"Dog":"🐶","Wolf":"🐺"}
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, caption=f.name)
            label, conf = processor.predict_cnn(img)
            st.markdown(f"""
            <div class="result-card">
                <h3>{emoji_map.get(label,label)} {label}</h3>
                <p>Akurasi: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; font-size:12px; color:#555;'>
Dashboard by Layla Ahmady Hsb 🔥 | Simple, Unik & Interaktif
</p>
""", unsafe_allow_html=True)
