# Disable GPU
import os

import tensorflow as tf
from tensorflow import keras

from com.seregy77.evnn.neural.network import Network
from com.seregy77.evnn.neural.utils import normalize_images, one_hot_encode
from com.seregy77.evnn.spea2.network_config import NetworkConfig
from com.seregy77.evnn.spea2.spea2 import Spea2
from com.seregy77.evnn.spea2.spea2_config import Spea2Config

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.compat.v1.disable_eager_execution()

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
result = Spea2(spea2_config, network_config).execute()

fashion_mnist = keras.datasets.fashion_mnist
mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = normalize_images(train_images, test_images)
train_labels, test_labels = one_hot_encode(train_labels, test_labels)

network = Network()
network.assign_custom_weights(result[0].weights)
network.compile()
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)
