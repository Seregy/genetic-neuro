from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential

from com.seregy77.evnn.neural import optimizer
from com.seregy77.evnn.neural.layer import Layer
from com.seregy77.evnn.neural.network import Network
from com.seregy77.evnn.neural.utils import normalize_images, one_hot_encode
from sklearn.preprocessing import StandardScaler


# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MAX_EPOCHS = 500
ACCURACY_THRESHOLD = 0.9

tf.compat.v1.disable_eager_execution()

fashion_mnist = keras.datasets.fashion_mnist
# mnist = keras.datasets.mnist
boston_housing = keras.datasets.boston_housing

(train_images, train_labels), (test_images, test_labels) = boston_housing.load_data()
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
# train_images, test_images = normalize_images(train_images, test_images)
# train_labels, test_labels = one_hot_encode(train_labels, test_labels)

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# model = Sequential()
# model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
# model.add(Dense(1, kernel_initializer='normal'))
# # Compile model
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(train_images, train_labels, epochs=150, batch_size=len(train_images),
#                                   validation_split=0.2)

mean = train_images.mean(axis=0)
train_images -= mean
std = train_images.std(axis=0)
train_images /= std

test_images -= mean
test_images /= std

network = Network(output_classes=1)
network.compile(optimizer=optimizer.NESTEROV, loss='mean_squared_error', metrics=['mean_absolute_error'])
history = network.fit(train_images, train_labels, epochs=100000, accuracy_stop_value=3)
test_mse_score, test_mae_score = network.evaluate(test_images, test_labels)

# print('\nTest accuracy:', test_acc)
#
# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# axes = plt.gca()
# axes.set_ylim([0.0, 1])
# plt.show()




# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
