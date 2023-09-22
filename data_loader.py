import numpy as np
from elder_futhark import runes
from tensorflow import keras


def get_data_sets(pix_min, pix_max, n, inp_size):
    """load arguments to get_data_sets to input custom datasets - [def get_data_sets(n, inp_size):]"""
    examples = [np.random.randint(pix_min, pix_max+1, size=inp_size) for _ in range(n)]
    # (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # examples = list(X_train)

    def training_set():
        while True:
            i = np.random.choice(len(examples))
            yield examples[i]

    def evaluation_set():
        while True:
            i = np.random.choice(len(examples))
            yield examples[i]

    return training_set, evaluation_set
