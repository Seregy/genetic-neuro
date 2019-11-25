import numpy as np


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

    def weights_as_vector(self):
        weights = self._weights
        vector = []
        for i in range(len(weights)):
            nested_array = weights[i]
            for j in range(len(nested_array)):
                vector.extend(np.array(nested_array[j]).flatten())

        return vector

    def weights_from_vector(self, vector,  layers_config):
        layers_amount = len(layers_config) - 1
        current_index = 0

        new_weights = []

        for i in range(layers_amount):
            matrix_i = layers_config[i]
            matrix_j = layers_config[i + 1]
            next_index = current_index + matrix_i * matrix_j

            weights_vector = vector[current_index:next_index]
            current_index = next_index

            weights_matrix = np.reshape(weights_vector, (matrix_i, matrix_j))

            next_index = current_index + matrix_j
            bias_vector = vector[current_index:next_index]
            current_index = next_index

            bias_matrix = np.reshape(bias_vector, (matrix_j,))

            new_weights.append([weights_matrix, bias_matrix])

        self.weights = new_weights

    def __str__(self) -> str:
        return "First: {}, Second: {}, Weights: {}".format(self._first_objective, self._second_objective, self._weights)
