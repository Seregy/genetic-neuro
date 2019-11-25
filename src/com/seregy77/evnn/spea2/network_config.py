class NetworkConfig:
    _layers = None
    _max_eight = None
    _min_weight = None
    _max_bias = None
    _min_bias = None

    def __init__(self,
                 layers,
                 max_weight,
                 min_weight,
                 max_bias,
                 min_bias):
        self._layers = layers
        self._max_eight = max_weight
        self._min_weight = min_weight
        self._max_bias = max_bias
        self._min_bias = min_bias

    @property
    def layers(self):
        return self._layers

    @property
    def max_weight(self):
        return self._max_eight

    @property
    def min_weight(self):
        return self._min_weight

    @property
    def max_bias(self):
        return self._max_bias

    @property
    def min_bias(self):
        return self._min_bias
