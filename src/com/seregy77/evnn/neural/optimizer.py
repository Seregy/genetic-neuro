from tensorflow_core.python.keras.optimizer_v2.adadelta import Adadelta
from tensorflow_core.python.keras.optimizer_v2.adagrad import Adagrad
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_core.python.keras.optimizer_v2.rmsprop import RMSprop

GD = SGD(learning_rate=0.01)
ADA_GRAD = Adagrad(learning_rate=0.01)
RMS_PROP = RMSprop(learning_rate=0.01)
ADA_DELTA = Adadelta(learning_rate=0.01)
ADAM = Adam(learning_rate=0.01)
NESTEROV = SGD(learning_rate=0.01, nesterov=True)
