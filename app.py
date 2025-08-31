import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "best.pt")
    if not os.path.exists(model_path):
        st.error("❌ Model tidak ditemukan. Pastikan file `model/best.pt` ada di repo.")
        st.stop()
    return YOLO(model_path)

# Mapping class → label Indo
classes = {
    "anger": "Marah",
    "contempt": "Menghina",
    "disgust": "Jijik",
    "fear": "Takut",
    "happiness": "Bahagia",
    "neutrality": "Netral",
    "sadness": "Sedih",
    "surprise": "Terkejut"
}

# === Fungsi untuk ambil gambar ikon ===
def get_class_image(class_name):
    static_dir = "static/images"
    exts = ["jpg", "jpeg", "png"]
    for ext in exts:
        path = os.path.join(static_dir, f"{class_name}.{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(static_dir, "default.jpg")  # fallback

# === Sidebar Navigasi (Dropdown) ===
page = st.sidebar.selectbox("📌 Pilih Halaman", ["Beranda", "Deteksi Foto", "Deteksi Realtime"])

# ===========================
# HALAMAN 1 - BERANDA
# ===========================
if page == "Beranda":
    st.title("🎭 Deteksi Emosi Wajah")
    st.write("Aplikasi ini dapat mengenali ekspresi wajah menggunakan Convolutional Neural Network (CNN).")
    
    st.subheader("✨ Emosi yang dapat dikenali:")
    cols = st.columns(4)
    for i, (eng, indo) in enumerate(classes.items()):
        with cols[i % 4]:
            img_path = get_class_image(eng)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((200, 200))
                st.image(img, caption=f"{indo} ({eng})", use_container_width=True)
            except:
                st.write(f"{indo} ({eng})")

# ===========================
# HALAMAN 2 - DETEKSI FOTO
# ===========================
elif page == "Deteksi Foto":
    st.title("📷 Deteksi Emosi dari Foto")
    uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
            ("dan cek permission kamera di browser.")


