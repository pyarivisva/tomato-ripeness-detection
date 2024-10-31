# PCD mendeteksi kematangan tomat dengan metode rgb dimana ekstraksi fitur tidak memperhitungkan 
# apakah piksel tersebut termasuk dalam objek tomat atau tidak, 
# sehingga nilai rata-rata RGB bisa terpengaruh oleh latar belakang atau bagian gambar yang tidak relevan.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Akuisisi Gambar
def acquire_image(img_path):
    img = Image.open(img_path)
    return img

# Path ke gambar tomat yg ingin dideteksi
img_path = 'D:/Source Code/tomato-ripeness-detection/validation/validation 11.jpeg' 

# 2. Preprocessing
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize gambar untuk mengurangi ukuran
    img = img.convert('RGB')  # Pastikan gambar dalam format RGB
    return img

# 3. Segmentasi
def segment_image(img):
    # Untuk metode RGB, kita tidak melakukan segmentasi yang rumit.
    # Kita langsung mengubah gambar menjadi array numpy.
    img_array = np.array(img)
    return img_array

# 4. Ekstraksi Fitur (Rata-rata RGB)
def extract_rgb_features(img_array):
    avg_r = np.mean(img_array[:, :, 0])  # Rata-rata Red
    avg_g = np.mean(img_array[:, :, 1])  # Rata-rata Green
    avg_b = np.mean(img_array[:, :, 2])  # Rata-rata Blue
    return avg_r, avg_g, avg_b

# 5. Klasifikasi
def classify_tomato_by_rgb(avg_r, avg_g, avg_b):
    if avg_r > 150 and avg_g < 100:  # Merah dominan
        return "Matang"
    elif avg_g > avg_r and avg_g > avg_b:  # Hijau dominan
        return "Mentah"
    else:  # Gabungan merah dan hijau
        return "Setengah Matang"

# 6. Fungsi Utama
def main(img_path):
    # Tahap 1: Akuisisi Gambar
    img = acquire_image(img_path)

    # Tahap 2: Preprocessing
    img = preprocess_image(img)

    # Tahap 3: Segmentasi
    img_array = segment_image(img)

    # Tahap 4: Ekstraksi Fitur
    avg_r, avg_g, avg_b = extract_rgb_features(img_array)
    print(f"Rata-rata R: {avg_r:.2f}, G: {avg_g:.2f}, B: {avg_b:.2f}")

    # Tahap 5: Klasifikasi
    result = classify_tomato_by_rgb(avg_r, avg_g, avg_b)
    print(f"Hasil Klasifikasi: {result}")

    # Menampilkan gambar
    plt.imshow(img)
    plt.axis('off')  # Matikan sumbu
    plt.show()

# Jalankan fungsi utama
main(img_path)
