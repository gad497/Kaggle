import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)
image_path = "car_illus.jpg"
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])
img = tf.squeeze(image).numpy()
plt.figure(figsize=(18,6))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('original image')

kernel = tf.constant([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape,1,1])
kernel = tf.cast(kernel, dtype=tf.float32)

conv_fn = tf.nn.conv2d

image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME'
)
plt.subplot(1, 4, 2)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('convolution')

relu_fn = tf.nn.relu

image_detect = relu_fn(image_filter)
plt.subplot(1, 4, 3)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('convolution + relu')

image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2,2),
    pooling_type='MAX',
    strides=(2,2),
    padding='SAME'
)

plt.subplot(1,4,4)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('conv + relu + maxpool')
plt.show()
