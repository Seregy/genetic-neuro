import copy
from unittest import TestCase
import numpy as np

from com.seregy77.evnn.spea2.individual import Individual


class TestIndividual(TestCase):

    def test_(self):
        individual = Individual(self.init_weights())
        vector = individual.weights_as_vector()

        individual_from_vector = Individual([])
        individual_from_vector.weights_from_vector(vector, [784, 50, 50, 10])

        close = False
        self.assertEqual(len(individual.weights), len(individual_from_vector.weights))

        for i in range(len(individual.weights)):
            for j in range(len(individual.weights[i])):
                close = np.allclose(individual.weights[i][j], individual_from_vector.weights[i][j])

                if not close:
                    self.fail("One of the arrays differs")

        self.assertTrue(close)

    def init_weights(self):
        layers_config = [784, 50, 50, 10]
        MAX_WEIGHT = 0.3 * 1.0
        MIN_WEIGHT = -0.3 * 1.0
        MAX_BIAS = 0.3 * 1.0
        MIN_BIAS = -0.3 * 1.0
        layer_amount = len(layers_config)
        weights = []
        for i in range(layer_amount - 1):
            # Weights
            weights_with_biases = [
                np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, (layers_config[i], layers_config[i + 1])),
                np.random.uniform(MIN_BIAS, MAX_BIAS, (layers_config[i + 1],))]
            weights.append(weights_with_biases)

        return weights