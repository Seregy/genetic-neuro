from tensorflow_core.python.keras.utils import np_utils


def normalize_images(*values):
    normalized_values = []
    for i in range(len(values)):
        normalized = values[i] / 255.0
        normalized_values.append(normalized)

    return tuple(normalized_values)


def one_hot_encode(*labels, class_amount=10):
    encoded_labels = []
    for i in range(len(labels)):
        one_hot_encoded = np_utils.to_categorical(labels[i], class_amount)
        encoded_labels.append(one_hot_encoded)

    return tuple(encoded_labels)
