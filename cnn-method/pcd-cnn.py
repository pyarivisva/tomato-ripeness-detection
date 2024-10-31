import tensorflow as tf  # Mengimpor TensorFlow untuk membangun model
from tensorflow import keras  # Mengimpor keras dari TensorFlow
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator  # Mengimpor ImageDataGenerator untuk augmentasi gambar
import numpy as np  # Mengimpor NumPy untuk manipulasi array
import matplotlib.pyplot as plt  # Mengimpor Matplotlib untuk visualisasi
from sklearn.model_selection import train_test_split  # Mengimpor fungsi untuk membagi data menjadi train dan test
import pandas as pd  # Mengimpor Pandas untuk manipulasi DataFrame
import os  # Mengimpor OS untuk operasi file dan direktori

# Fungsi untuk membuat model CNN
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # Lapisan konvolusi pertama
        tf.keras.layers.MaxPooling2D(2, 2),  # Max pooling untuk mengurangi dimensi
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Lapisan konvolusi kedua
        tf.keras.layers.MaxPooling2D(2, 2),  # Max pooling kedua
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Lapisan konvolusi ketiga
        tf.keras.layers.MaxPooling2D(2, 2),  # Max pooling ketiga
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Lapisan konvolusi keempat
        tf.keras.layers.MaxPooling2D(2, 2),  # Max pooling keempat
        tf.keras.layers.Flatten(),  # Mengubah hasil ke dalam bentuk vektor
        tf.keras.layers.Dense(512, activation='relu'),  # Lapisan dense dengan 512 neuron
        tf.keras.layers.Dense(3, activation='softmax')  # Lapisan output dengan 3 kelas: Setengah Matang, Mentah, Matang
    ])

    # Mengkompilasi model dengan optimizer Adam dan loss categorical_crossentropy
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Persiapan data
data_dir = 'processed_images'  # Direktori untuk dataset yang telah diproses

# Ambil list filenames dan labels dari struktur folder
filenames = []  # Untuk menyimpan path file gambar
labels = []    # Untuk menyimpan label kategori gambar
class_names = ["Matang", "Mentah", "Setengah Matang"]  # Nama kelas dari gambar
for class_name in class_names:
    class_folder = os.path.join(data_dir, class_name)  # Menggabungkan path folder kelas
    for file in os.listdir(class_folder):
        filenames.append(os.path.join(class_name, file))  # Menambahkan path file ke list
        labels.append(class_name)  # Menambahkan label ke list

# Bagi data menjadi train dan test (80% untuk train, 20% untuk test)
X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.2, random_state=42)

# Buat DataFrame untuk data latih dan uji
train_df = pd.DataFrame({'filename': X_train, 'class': y_train})  # DataFrame untuk data latih
test_df = pd.DataFrame({'filename': X_test, 'class': y_test})  # DataFrame untuk data uji

# Buat ImageDataGenerator dan generator data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi pixel ke rentang 0-1
    rotation_range=40,  # Rentang rotasi gambar
    width_shift_range=0.2,  # Perubahan lebar gambar
    height_shift_range=0.2,  # Perubahan tinggi gambar
    shear_range=0.2,  # Geser gambar
    zoom_range=0.2,  # Zoom gambar
    horizontal_flip=True,  # Membalik gambar secara horizontal
    fill_mode='nearest'  # Metode pengisian piksel kosong
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Hanya normalisasi untuk data uji

# Generator data latih
train_generator = train_datagen.flow_from_dataframe(
    train_df,  # DataFrame untuk data latih
    directory=data_dir,  # Direktori gambar
    x_col='filename',  # Kolom untuk path file
    y_col='class',  # Kolom untuk label kelas
    target_size=(150, 150),  # Ukuran target gambar
    batch_size=32,  # Ukuran batch
    class_mode='categorical'  # Tipe kelas untuk multi-kelas
)

# Generator data validasi
validation_generator = test_datagen.flow_from_dataframe(
    test_df,  # DataFrame untuk data uji
    directory=data_dir,  # Direktori gambar
    x_col='filename',  # Kolom untuk path file
    y_col='class',  # Kolom untuk label kelas
    target_size=(150, 150),  # Ukuran target gambar
    batch_size=32,  # Ukuran batch
    class_mode='categorical'  # Tipe kelas untuk multi-kelas
)

# Buat dan latih model
model = create_model()  # Membuat model
history = model.fit(
    train_generator,  # Generator data latih
    epochs=10,  # Jumlah epoch
    validation_data=validation_generator  # Generator data validasi
)

# Evaluasi model
test_loss, test_acc = model.evaluate(validation_generator)  # Menghitung loss dan akurasi pada data validasi
print('Test accuracy:', test_acc)  # Menampilkan akurasi model

# Simpan model
model.save('model_tomat.h5')  # Menyimpan model ke file
