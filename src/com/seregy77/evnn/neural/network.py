from tensorflow import keras
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD

from com.seregy77.evnn.neural.early_stop import EarlyAccuracyStop, EarlyMaeStop
from com.seregy77.evnn.neural.layer import Layer


class Network:
    _model = None

    def __init__(self,
                 input_shape=(28, 28),
                 hidden_layers=None,
                 output_classes=10,
                 trainable=True):
        if hidden_layers is None:
            hidden_layers = []

        input_layer = keras.layers.Dense(16, input_dim=13, activation='relu')
        output_layer = keras.layers.Dense(output_classes, activation='linear')

        keras_layers = []
        for layer in hidden_layers:
            keras_layer = keras.layers.Dense(layer.node_amount, activation=layer.activation_function)
            keras_layers.append(keras_layer)

        model = keras.Sequential([
            input_layer,
            *keras_layers,
            output_layer
        ])
        model.trainable = trainable

        self._model = model

    def compile(self, optimizer=None, loss='categorical_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']

        if optimizer is None:
            optimizer = SGD(learning_rate=0.01)

        self._model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    def fit(self, train_images, train_labels, epochs=20000, batch_size=None, accuracy_stop_value=0.85):
        if batch_size is None:
            batch_size = len(train_images)

        callbacks = [EarlyMaeStop(value=accuracy_stop_value, verbose=1)]

        history = self._model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                                  validation_split=0.2)
        return history

    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self._model.evaluate(test_images, test_labels, verbose=2)
        return test_loss, test_acc

    def assign_custom_weights(self, weights):
        model = self._model

        for i in range(len(weights)):
            new_weights = [weights[i].weights, weights[i].biases]
            model.layers[i].set_weights(new_weights)
