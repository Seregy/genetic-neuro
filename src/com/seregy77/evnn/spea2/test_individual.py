from unittest import TestCase

import numpy as np

from com.seregy77.evnn.spea2.individual import Individual
from com.seregy77.evnn.spea2.network_parameter import LayerParameter


def init_weights():
    layers_config = [784, 50, 50, 10]
    max_weight = 0.3 * 1.0
    min_weight = -0.3 * 1.0
    max_bias = 0.3 * 1.0
    min_bias = -0.3 * 1.0
    layer_amount = len(layers_config)
    layer_params = []
    for i in range(layer_amount - 1):
        # Weights
        weights = np.random.uniform(min_weight, max_weight, (layers_config[i], layers_config[i + 1]))
        biases = np.random.uniform(min_bias, max_bias, (layers_config[i + 1],))
        layer_param = LayerParameter(weights,
                                     biases)
        layer_params.append(layer_param)

    return layer_params


class TestIndividual(TestCase):

    def cross(self):
        individual = Individual(init_weights())
        individual2 = Individual(init_weights())

        (new_individual1, new_individual2) = individual.cross(individual2)

        self.assertNotEqual(new_individual1, new_individual2)
        self.assertNotEqual(individual, new_individual1)
        self.assertNotEqual(individual2, new_individual2)
