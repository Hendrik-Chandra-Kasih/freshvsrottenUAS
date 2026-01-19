from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

CLASS_NAMES = ['Fresh', 'Rotten']
TARGET_SIZE = (150, 150)

CONF_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.40

print("Memuat model...")
model = load_model('model_uas1.h5')
print("Model siap!")

def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Fresh & Rotten Detection</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #f0f2f5;
    padding: 20px;
}
.container {
    max-width: 700px;
    margin: auto;
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
h1 {
    text-align: center;
}
button {
    padding: 10px 16px;
    margin: 5px;
    border: none;
    border-radius: 6px;
    background: #3498db;
    color: white;
    cursor: pointer;
}
.section { display: none; }
.section.active { display: block; }

#drop-area {
    margin-top: 15px;
    padding: 30px;
    border: 2px dashed #3498db;
    border-radius: 10px;
    text-align: center;
    cursor: pointer;
    background: #f9fcff;
    transition: 0.3s;
}
#drop-area.dragover {
    background: #eaf4ff;
    border-color: #1d6fa5;
}

#result {
    margin-top: 20px;
    padding: 15px;
    background: #e8f4fc;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

video {
    width: 320px;
    border-radius: 8px;
    background: black;
}
</style>
</head>

<body>
<div class="container">
<h1>📸 Fresh & Rotten Detection</h1>

<div style="text-align:center;">
    <button id="btn-upload">📤 Upload</button>
    <button id="btn-webcam">🎥 Webcam</button>
</div>

<div id="upload-section" class="section active">
    <div id="drop-area">
        <p>🖱️ Drag & Drop gambar di sini</p>
        <p>atau klik untuk memilih file</p>
    </div>
    <input type="file" id="file-input" accept="image/*" hidden>
    <button onclick="predictUpload()">Prediksi</button>
    <div id="upload-preview"></div>
</div>

<div id="webcam-section" class="section">
    <video id="video" autoplay muted></video><br>
    <button onclick="captureAndPredict()">Ambil & Prediksi</button>
    <div id="webcam-preview"></div>
</div>

<div id="result"></div>
</div>

<script>
const uploadSec = document.getElementById('upload-section');
const webcamSec = document.getElementById('webcam-section');

document.getElementById('btn-upload').onclick = () => {
    uploadSec.classList.add('active');
    webcamSec.classList.remove('active');
};

document.getElementById('btn-webcam').onclick = () => {
    uploadSec.classList.remove('active');
    webcamSec.classList.add('active');
    startCamera();
};

let stream = null;
async function startCamera() {
    if (stream) return;
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('video').srcObject = stream;
}

// =======================
// DRAG & DROP
// =======================
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');

dropArea.addEventListener('click', () => fileInput.click());

dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith('image/')) {
        alert('Harus file gambar!');
        return;
    }

    fileInput.files = e.dataTransfer.files;
    predictUpload();
});

// =======================
// UPLOAD
// =======================
function predictUpload() {
    const file = fileInput.files[0];
    if (!file) {
        alert('Pilih gambar dulu!');
        return;
    }

    const reader = new FileReader();
    reader.onload = async e => {
        document.getElementById('upload-preview').innerHTML =
            `<img src="${e.target.result}" style="max-width:100%; margin-top:10px;">`;

        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        showResult(data);
    };
    reader.readAsDataURL(file);
}

// =======================
// WEBCAM
// =======================
async function captureAndPredict() {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg');

    document.getElementById('webcam-preview').innerHTML =
        `<img src="${dataUrl}" style="max-width:100%; margin-top:10px;">`;

    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    });
    const data = await res.json();
    showResult(data);
}

// =======================
// RESULT
// =======================
function showResult(data) {
    const resDiv = document.getElementById('result');

    if (data.error) {
        resDiv.innerHTML = `<p style="color:red;">❌ ${data.error}</p>`;
        return;
    }

    if (data.class === 'Not Recognized') {
        resDiv.innerHTML = `
            <p>❌ <strong>Not Recognized</strong></p>
            <p>Gambar bukan Buah</p>
        `;
        return;
    }

    resDiv.innerHTML = `
        <p>🎯 Prediksi: <strong>${data.class}</strong></p>
        <p>🔍 Confidence: <strong>${data.confidence}%</strong></p>
    `;
}
</script>
</body>
</html>
'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            img = Image.open(request.files['file'].stream)

        elif request.is_json:
            data = request.get_json()
            img_data = data['image'].split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(img_data)))

        else:
            return jsonify({'error': 'Tidak ada data gambar'}), 400

        processed = preprocess_image(img)
        pred = model.predict(processed)[0]

        top1 = float(np.max(pred))
        top2 = float(np.sort(pred)[-2])
        margin = top1 - top2

        if top1 < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
            return jsonify({'class': 'Not Recognized'})

        idx = int(np.argmax(pred))
        return jsonify({
            'class': CLASS_NAMES[idx],
            'confidence': round(top1 * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
