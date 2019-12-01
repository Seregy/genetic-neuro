class LayerParameter:
    _weights = None
    _biases = None

    def __init__(self, weights, biases):
        self._weights = weights
        self._biases = biases

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases
