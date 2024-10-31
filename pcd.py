import tensorflow as tf
# from tensorflow import keras
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Fungsi untuk membuat model CNN
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 kelas: Setengah Matang, Mentah, Matang
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Persiapan data
data_dir = 'processed_images'

# Ambil list filenames dan labels dari struktur folder
filenames = []
labels = []
class_names = ["Matang", "Mentah", "Setengah Matang"]
for class_name in class_names:
    class_folder = os.path.join(data_dir, class_name)
    for file in os.listdir(class_folder):
        filenames.append(os.path.join(class_name, file))
        labels.append(class_name)

# Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.2, random_state=42)

# Buat DataFrame untuk data latih dan uji
train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
test_df = pd.DataFrame({'filename': X_test, 'class': y_test})

# Buat ImageDataGenerator dan generator data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Generator data latih
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_dir,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Generator data validasi
validation_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=data_dir,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Buat dan latih model
model = create_model()
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluasi model
test_loss, test_acc = model.evaluate(validation_generator)
print('Test accuracy:', test_acc)

# Simpan model
model.save('model_tomat.h5')
