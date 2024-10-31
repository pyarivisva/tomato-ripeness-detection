import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ambil list nama file dan label dari struktur folder
filenames = []  # Untuk menyimpan path file gambar
labels = []    # Untuk menyimpan label kategori gambar
data_dir = 'archive'  # Ganti dengan path direktori dataset
class_names = ["Matang", "Mentah", "Setengah Matang"]  # Nama kelas dari gambar

# Mengambil nama file dan label dari setiap kelas di folder dataset
for class_name in class_names:
    class_folder = os.path.join(data_dir, class_name)  # Menggabungkan path folder kelas
    for file in os.listdir(class_folder):
        file_path = os.path.join(class_folder, file)  # Menggabungkan path file
        if os.path.isfile(file_path):  # Pastikan itu file, bukan folder
            filenames.append(file_path)  # Menambahkan path file ke list
            labels.append(class_name)  # Menambahkan label ke list

# Fungsi untuk meresize gambar menggunakan Pillow
def resize_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)  # Membuka gambar
    img_resized = img.resize(target_size)  # Meresize gambar ke ukuran target
    return np.array(img_resized)  # Mengembalikan gambar sebagai array NumPy

# Fungsi untuk normalisasi pixel (0-1) menggunakan TensorFlow
def normalize_image(image_array):
    return image_array / 255.0  # Mengembalikan nilai pixel dalam rentang 0-1

# Fungsi untuk augmentasi rotasi menggunakan TensorFlow
def augment_image(image_array, rotation_range=40):
    datagen = ImageDataGenerator(rotation_range=rotation_range)  # Membuat objek augmentasi
    image_array = image_array.reshape((1,) + image_array.shape)  # Mengubah bentuk array gambar
    augmented_image = next(datagen.flow(image_array, batch_size=1))[0]  # Menghasilkan gambar yang diaugmentasi
    return augmented_image  # Mengembalikan gambar yang telah diaugmentasi

# Contoh pipeline preprocessing
def preprocess_image(image_path):
    # Resize gambar
    resized_img = resize_image(image_path)

    # Normalisasi pixel
    normalized_img = normalize_image(resized_img)

    # Konversi kembali ke uint8 sebelum augmentasi
    normalized_img_uint8 = (normalized_img * 255).astype(np.uint8)

    # Augmentasi rotasi
    augmented_img = augment_image(normalized_img_uint8)

    # Mengembalikan hasil akhir setelah preprocessing
    return augmented_img

# Buat folder output untuk setiap kelas jika belum ada
output_dir = 'processed_images'  # Direktori untuk menyimpan gambar yang telah diproses
for class_name in class_names:
    class_output_dir = os.path.join(output_dir, class_name)  # Path untuk setiap kelas
    if not os.path.exists(class_output_dir):  # Cek jika folder belum ada
        os.makedirs(class_output_dir)  # Membuat folder kelas

# Contoh penggunaan
for image_path, label in zip(filenames, labels):  # Menggunakan filenames dan labels yang diambil dari folder
    processed_image = preprocess_image(image_path)  # Memproses gambar

    # Mengubah dari BGR ke RGB
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Simpan gambar hasil preprocessing sesuai dengan label
    output_path = os.path.join(output_dir, label, f'processed_{os.path.basename(image_path)}')  # Path untuk menyimpan gambar yang diproses
    
    # Menggunakan cv2 untuk menyimpan gambar hasil preprocessing
    cv2.imwrite(output_path, processed_image_rgb)  # Menyimpan gambar

print("Preprocessing selesai dan gambar disimpan.")  # Menampilkan pesan bahwa proses telah selesai
