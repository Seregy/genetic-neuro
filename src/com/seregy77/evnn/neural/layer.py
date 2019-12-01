class Layer:
    _node_amount = None
    _activation_function = None

    def __init__(self, node_amount, activation):
        self._node_amount = node_amount
        self._activation_function = activation

    @property
    def node_amount(self):
        return self._node_amount

    @property
    def activation_function(self):
        return self._activation_function
