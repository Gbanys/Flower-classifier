from ml_workflow.preprocess_from_folders import create_labels, extract_images
from sklearn.model_selection import train_test_split
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple

def train_test_validation_split(images, labels):

    X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify=labels, test_size=0.10)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.10)

    X_train = X_train / 255
    X_valid = X_valid / 255
    X_test = X_test / 255

    ModelData = namedtuple("ModelData", ["x_train", "x_valid", "x_test", "y_train", "y_valid", "y_test"])

    ml_data = ModelData(X_train, X_valid, X_test, y_train, y_valid, y_test)

    return ml_data


def train():

    labels = create_labels()
    images = extract_images()
    ml_data = train_test_validation_split(images, labels)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[60, 60, 3]),
        keras.layers.Dense(200, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

    sgd = keras.optimizers.SGD(learning_rate=0.01)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    history = model.fit(
        ml_data.x_train,
        ml_data.y_train,
        epochs=200,
        validation_data=(ml_data.x_valid, ml_data.y_valid)
    )

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    return model
