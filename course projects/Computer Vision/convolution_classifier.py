import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18,
titlepad=10)
plt.rc('image', cmap='magma')
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
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False
)

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, size=(299, 299))
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (
    ds_train.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)

base_model = InceptionV3(weights="imagenet", include_top=True, input_shape=(299, 299, 3))
# base_model.summary()

pretrained_base = Model(inputs=base_model.input, outputs=base_model.get_layer("mixed10").output)
pretrained_base.trainable = False
# pretrained_base.summary()

model = tf.keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy']
)
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=3
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss','val_loss']].plot()
history_df.loc[:, ['binary_accuracy','val_binary_accuracy']].plot()
plt.show()