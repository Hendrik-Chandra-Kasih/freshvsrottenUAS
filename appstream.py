import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ================= CONFIG =================
CLASS_NAMES = ['Fresh', 'Rotten']
TARGET_SIZE = (150, 150)
CONF_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.40

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner=True)
def load_model_once():
    return load_model('model_uas1.h5')

model = load_model_once()

# ================= PREPROCESS =================
def preprocess_image(img: Image.Image):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ================= PREDICT =================
def predict(img: Image.Image):
    processed = preprocess_image(img)
    pred = model.predict(processed)[0]

    top1 = float(np.max(pred))
    top2 = float(np.sort(pred)[-2])
    margin = top1 - top2

    if top1 < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        return 'Not Recognized', None

    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx], round(top1 * 100, 2)

# ================= UI =================
st.set_page_config(page_title="Fresh vs Rotten Detection", layout="centered")
st.title("🍎 Fresh vs Rotten Detection")

st.write("Pilih metode input gambar:")

input_mode = st.radio(
    "Metode Input",
    ("📂 Upload Gambar", "📸 Ambil dari Kamera"),
    horizontal=True
)

img = None

# ===== UPLOAD MODE =====
if input_mode == "📂 Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar buah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar diupload", use_column_width=True)

# ===== CAMERA MODE =====
elif input_mode == "📸 Ambil dari Kamera":
    camera_file = st.camera_input("Ambil gambar buah")

    if camera_file is not None:
        img = Image.open(camera_file)
        st.image(img, caption="Gambar dari kamera", use_column_width=True)

# ================= ACTION =================
if img is not None:
    if st.button("🔍 Prediksi"):
        label, conf = predict(img)

        if label == "Not Recognized":
            st.error("❌ Objek tidak dikenali sebagai buah.")
        else:
            st.success(f"✅ Prediksi: **{label}**")
            st.info(f"Confidence: **{conf}%**")

st.caption("CNN Image Classification • Streamlit • UAS")
