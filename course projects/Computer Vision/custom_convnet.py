import os, warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

warnings.filterwarnings('ignore')

ds_train = image_dataset_from_directory(
    'car or truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128,128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True
)
ds_valid = image_dataset_from_directory(
    'car or truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128,128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False
)

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32),
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)

model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[128,128,3]),
    layers.MaxPool2D(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=3
)

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['binary_accuracy','val_binary_accuracy']].plot()
plt.show()