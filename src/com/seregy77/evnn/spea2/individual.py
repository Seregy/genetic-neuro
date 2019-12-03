import random

import numpy as np

from com.seregy77.evnn.spea2.network_parameter import LayerParameter


class Individual:
    _weights = []
    _fitness = None
    _first_objective = None
    _second_objective = None

    def __init__(self, weights):
        self._weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @property
    def first_objective(self):
        return self._first_objective

    @first_objective.setter
    def first_objective(self, value):
        self._first_objective = value

    @property
    def second_objective(self):
        return self._second_objective

    @second_objective.setter
    def second_objective(self, value):
        self._second_objective = value

    def cross(self, second_individual):
        first_child_weights = []
        second_child_weights = []

        layer_amount = len(self.weights)
        layer_index = random.randrange(layer_amount)

        for i in range(layer_amount):
            if i < layer_index:
                first_child_weights.append(self.weights[i])
                second_child_weights.append(second_individual.weights[i])
            elif i > layer_index:
                first_child_weights.append(second_individual.weights[i])
                second_child_weights.append(self.weights[i])
            else:
                first_individual_layer = self.weights[layer_index]
                second_individual_layer = second_individual.weights[layer_index]

                first_new_weights = first_individual_layer.weights
                first_new_biases = first_individual_layer.biases
                second_new_weights = second_individual_layer.weights
                second_new_biases = first_individual_layer.biases

                swap_weights = random.randrange(2) == 0
                if swap_weights:
                    first_weights = first_individual_layer.weights
                    first_weights_vector = first_weights.ravel()
                    second_weights = second_individual_layer.weights
                    second_weights_vector = second_weights.ravel()
                    weights_length = len(first_weights_vector)
                    random_vector_point = random.randrange(weights_length)
                    first_new_vector = np.empty(shape=(weights_length,))
                    second_new_vector = np.empty(shape=(weights_length,))

                    for j in range(weights_length):
                        if j < random_vector_point:
                            first_new_vector[j] = first_weights_vector[j]
                            second_new_vector[j] = second_weights_vector[j]
                        else:
                            first_new_vector[j] = second_weights_vector[j]
                            second_new_vector[j] = first_weights_vector[j]

                    first_new_weights = np.reshape(first_new_vector, first_weights.shape)
                    second_new_weights = np.reshape(second_new_vector, second_weights.shape)

                else:
                    biases_length = len(first_individual_layer.biases)
                    random_point = random.randrange(biases_length)
                    first_new_biases = np.empty(shape=(biases_length,))
                    second_new_biases = np.empty(shape=(biases_length,))

                    for j in range(biases_length):
                        if j < random_point:
                            first_new_biases[j] = first_individual_layer.biases[j]
                            second_new_biases[j] = second_individual_layer.biases[j]
                        else:
                            first_new_biases[j] = second_individual_layer.biases[j]
                            second_new_biases[j] = first_individual_layer.biases[j]

                first_child_weights.append(LayerParameter(first_new_weights, first_new_biases))
                second_child_weights.append(LayerParameter(second_new_weights, second_new_biases))

        return Individual(first_child_weights), Individual(second_child_weights)

    def mutate(self, min_weight, max_weight, min_bias, max_bias):
        new_individual = Individual(self.weights)

        layer_index = random.randrange(len(self.weights))
        random_layer = new_individual.weights[layer_index]

        mutate_weights = random.randrange(2) == 0

        if mutate_weights:
            weights = random_layer.weights
            weights_vector = weights.ravel()
            random_index = random.randrange(len(weights_vector))
            weights_vector[random_index] = random.uniform(min_weight, max_weight)
        else:
            biases = random_layer.biases
            random_index = random.randrange(len(biases))
            biases[random_index] = random.uniform(min_bias, max_bias)

        return new_individual

    def __str__(self) -> str:
        return "First: {}, Second: {}, Weights: {}".format(self._first_objective, self._second_objective, self._weights)
