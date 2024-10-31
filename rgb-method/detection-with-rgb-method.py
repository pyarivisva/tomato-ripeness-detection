import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. Akuisisi Gambar
def acquire_image(img_path):
    img = Image.open(img_path)
    return img

# 2. Preprocessing
# Resize gambar
def resize(img_array):
    return cv2.resize(img_array, (200, 200))

# Gaussian Blur
def gaussian_blur(img_array):
    return cv2.GaussianBlur(img_array, (5, 5), 0)

# CLAHE Equalization untuk mengatur kontras
def clahe_equalization(img_array):
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

# 3. Ekstraksi Fitur (Rata-rata RGB pada gambar setelah preprocessing)
def extract_rgb_features(img_array):
    r_channel = img_array[:, :, 0].flatten()
    g_channel = img_array[:, :, 1].flatten()
    b_channel = img_array[:, :, 2].flatten()
    
    avg_r = np.mean(r_channel)
    avg_g = np.mean(g_channel)
    avg_b = np.mean(b_channel)
    
    return avg_r, avg_g, avg_b

# 4. Klasifikasi (berdasarkan nilai rata-rata RGB)
def classify_tomato_by_rgb(avg_r, avg_g, avg_b):
    if avg_r > 150 and avg_g < 150:
        return "Matang"
    elif avg_g > avg_r and avg_g > avg_b:
        return "Mentah"
    else:
        return "Setengah Matang"

# 5.Memproses satu gambar dan mengklasifikasikannya
def process_image(img_path):
    img = acquire_image(img_path)
    img_array = np.array(img)  # Mengubah gambar ke array NumPy
    
    # Preprocessing dilakukan secara berurutan
    img_preprocessed = resize(img_array)           # Resize gambar
    img_preprocessed = gaussian_blur(img_preprocessed)   # Gaussian blur
    img_preprocessed = clahe_equalization(img_preprocessed)  # CLAHE equalization

    # Ekstraksi Fitur dari gambar yang sudah di-preprocess
    avg_r, avg_g, avg_b = extract_rgb_features(img_preprocessed)
    result = classify_tomato_by_rgb(avg_r, avg_g, avg_b)
    
    print(f"Gambar: {os.path.basename(img_path)}")
    print(f"Rata-rata R: {avg_r:.2f}, G: {avg_g:.2f}, B: {avg_b:.2f}")
    print(f"Hasil Klasifikasi: {result}\n")

    # Menampilkan hasil preprocessing
    plt.figure(figsize=(8, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Gambar Asli')

    plt.subplot(2, 2, 2)
    plt.imshow(resize(img_array))
    plt.axis('off')
    plt.title('Resize Gambar')

    plt.subplot(2, 2, 3)
    plt.imshow(gaussian_blur(resize(img_array)))
    plt.axis('off')
    plt.title('Gaussian Blurred')

    plt.subplot(2, 2, 4)
    plt.imshow(clahe_equalization(gaussian_blur(resize(img_array))))
    plt.axis('off')
    plt.title('CLAHE Equalized')
    
    plt.figtext(0.5, 0.01, f'Hasil Klasifikasi: {result}', ha='center')

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.show()

# 6. Fungsi Utama untuk memproses semua gambar di folder
def process_all_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Pastikan hanya file gambar
            img_path = os.path.join(folder_path, filename)
            process_image(img_path)

# Path ke folder berisi gambar tomat
folder_path = r'D:\Source Code\tomato-ripeness-detection\tomato'

# Jalankan fungsi untuk memproses semua gambar di folder
process_all_images_in_folder(folder_path)