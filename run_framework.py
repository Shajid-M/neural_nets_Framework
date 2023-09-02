from data_loader import get_data_sets
from nn_framework.framework import ANN
from nn_framework import layer
from nn_framework.activation import Tanh, Logistic, Relu
from nn_framework import error_fun
from nn_framework.autoencoder_viz import NNViz
PIXEL_VALUE_RANGE = (0, 256)
N_NODES = [1000]


def main():
    training_set, evaluation_set = get_data_sets()
    sample = next(training_set())
    n_pixels = sample.shape[0] * sample.shape[1]
    n_nodes = [n_pixels] + N_NODES + [n_pixels]

    model = []
    for i_layer in range(len(n_nodes) - 1):
        model.append(
            layer.Dense(
                n_nodes[i_layer],
                n_nodes[i_layer + 1],
                Tanh(),
                learning_rate=1e-2
            ))

    viz = NNViz(sample.shape)
    autoencoder = ANN(model=model, error_fun=error_fun.Abs, visualize=viz, pix_range=PIXEL_VALUE_RANGE)
    autoencoder.train(training_set)
    autoencoder.evaluate(evaluation_set)


if __name__ == '__main__':
    main()
