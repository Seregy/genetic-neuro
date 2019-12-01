import warnings

from tensorflow_core.python.keras.callbacks import Callback


class EarlyAccuracyStop(Callback):
    def __init__(self, monitor='val_accuracy', value=0.85, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
