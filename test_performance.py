import time
import tracemalloc
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('model_uas1.h5')


def preprocess_image(img: Image.Image):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


img = Image.open('apelh.jpg')  # pastikan ada gambar ini di folder kerja


tracemalloc.start()


start_time = time.time()

processed_img = preprocess_image(img)
pred = model.predict(processed_img)


end_time = time.time()


current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Waktu inferensi: {(end_time - start_time):.4f} detik")
print(f"Penggunaan memori puncak saat inferensi: {peak / 10**6:.4f} MB")
