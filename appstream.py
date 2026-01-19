import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Constants
CLASS_NAMES = ['Fresh', 'Rotten']
TARGET_SIZE = (150, 150)
CONF_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.40

# Load model sekali saat aplikasi dijalankan
@st.cache_resource(show_spinner=True)
def load_model_once():
    model = load_model('model_uas2.h5')
    return model

model = load_model_once()

# Fungsi preprocessing gambar
def preprocess_image(img: Image.Image):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Fungsi prediksi
def predict(img: Image.Image):
    processed = preprocess_image(img)
    pred = model.predict(processed)[0]

    top1 = float(np.max(pred))
    top2 = float(np.sort(pred)[-2])
    margin = top1 - top2

    if top1 < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        return 'Not Recognized', None

    idx = int(np.argmax(pred))
    confidence = round(top1 * 100, 2)
    return CLASS_NAMES[idx], confidence

# Streamlit UI
st.title("📸 Fresh & Rotten Detection")

st.write("Upload gambar buah untuk mengetahui apakah buah tersebut segar atau busuk.")

uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Gambar yang diupload', use_column_width=True)
        
        if st.button("Prediksi"):
            label, conf = predict(img)
            if label == 'Not Recognized':
                st.error("❌ Gambar tidak dikenali sebagai buah segar atau busuk.")
            else:
                st.success(f"🎯 Prediksi: **{label}**")
                st.info(f"🔍 Confidence: **{conf}%**")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.write("---")
st.write("Aplikasi dibuat menggunakan Streamlit dan model Keras yang telah dilatih.")
