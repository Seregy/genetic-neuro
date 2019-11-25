from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf

from com.seregy77.evnn.spea2.network_config import NetworkConfig
from com.seregy77.evnn.spea2.spea2 import Spea2
from com.seregy77.evnn.spea2.spea2_config import Spea2Config

EPOCHS = 300

#Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

fashion_mnist = keras.datasets.fashion_mnist
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

spea2_config = Spea2Config(population_size=25,
                           archive_size=25,
                           max_iterations=10,
                           crossover_probability=0.8,
                           mutation_probability=0.2)
network_config = NetworkConfig(layers=[784, 512, 512, 10],
                               max_weight=0.3,
                               min_weight=-0.3,
                               max_bias=0.3,
                               min_bias=-0.3)
# Load SPEA2 weights
individuals = Spea2(spea2_config, network_config).execute()

weights = individuals[0].weights

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

for i in range(len(weights)):
    model.layers[i + 1].set_weights(weights[i])

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=60000)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
axes = plt.gca()
axes.set_ylim([0.0,1])
plt.show()

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()