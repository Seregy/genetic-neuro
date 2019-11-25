# Disable GPU
import os
import tensorflow as tf

from com.seregy77.evnn.spea2.network_config import NetworkConfig
from com.seregy77.evnn.spea2.spea2 import Spea2
from com.seregy77.evnn.spea2.spea2_config import Spea2Config
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# import cProfile
#
# pr = cProfile.Profile()disable_eager_execution
# pr.enable()

tf.compat.v1.disable_eager_execution()

spea2_config = Spea2Config(population_size=100,
                           archive_size=10,
                           max_iterations=50,
                           crossover_probability=0.8,
                           mutation_probability=0.2)
network_config = NetworkConfig(layers=[784, 50, 50, 10],
                               max_weight=1,
                               min_weight=-1,
                               max_bias=1,
                               min_bias=-1)
result = Spea2(spea2_config, network_config).execute()

# pr.disable()
# # after your program ends
# pr.print_stats(sort="cumtime")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

weights = result[0].weights
for i in range(len(weights)):
    model.layers[i + 1].set_weights(weights[i])

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

