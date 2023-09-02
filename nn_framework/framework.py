import os
import numpy as np
import matplotlib.pyplot as plt


class ANN:

    def __init__(self, model=None, error_fun=None, visualize=None, pix_range=(-1, 1)):
        self.layers = model
        self.error_fun = error_fun
        self.visualize = visualize
        self.train_iter = int(1e8)
        self.eval_iter = int(1e6)
        self.pix_range = pix_range
        self.offset = self.pix_range[0]
        self.scale = (
            self.pix_range[1]
            - self.pix_range[0]
        )
        self.error_history = []
        self.viz_interval = int(1e5)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0
        self.report_path = "reports"
        self.report_name = "performance_history.png"
        try:
            os.mkdir("reports")
        except Exception:
            pass

    def train(self, training_set):
        for i_iter in range(self.train_iter):
            x = self.normalize(next(training_set()).ravel())
            y = self.forward_prop(x)
            error = self.error_fun.calc(x, y)
            error_d = self.error_fun.calc_d(x, y)
            rms = np.mean(error**2)**0.5
            self.error_history.append(rms)
            self.back_prop(error_d)
            if (i_iter + 1) % self.viz_interval == 0:
                self.report()
                self.visualize.render(self, x, f"{i_iter + 1:08d}")

    def evaluate(self, evaluation_set):
        for i_iter in range(self.eval_iter):
            x = self.normalize(next(evaluation_set()).ravel())
            y = self.forward_prop(x)
            error = self.error_fun.calc(x, y)
            rms = np.mean(error**2)**0.5
            self.error_history.append(rms)
            if (i_iter + 1) % self.viz_interval == 0:
                self.report()
                self.visualize.render(self, x, f"{i_iter + 1:08d}")

    def normalize(self, un_normalized):
        normalized = (un_normalized - self.offset) / self.scale - 0.5
        return normalized

    def de_normalize(self, normalized):
        de_normalized = (normalized + 0.5) * self.scale + self.offset
        return de_normalized

    def forward_prop(self, x):
        y = x.ravel()[np.newaxis, :]
        for i, layer in enumerate(self.layers):
            # print(f"passing through layer {i}")
            y = layer.forward_prop(y)
        return y.ravel()

    def forward_prop_to_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[:i_layer]:
            y = layer.forward_prop(y)
        return y.ravel()

    def forward_prop_from_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[i_layer:]:
            y = layer.forward_prop(y)
        return y.ravel()

    def back_prop(self, de_dy):
        for i_layer, layer in enumerate(self.layers[::-1]):
            de_dx = layer.back_prop(de_dy)
            de_dy = de_dx

    def report(self):
        n_bins = int(len(self.error_history)) // self.reporting_bin_size
        smoothed_history = []
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(self.error_history[i_bin * self.reporting_bin_size:
                                                               (i_bin + 1) * self.reporting_bin_size]))
        error_history = np.log10(np.array(smoothed_history) + 1e-10)
        y_min = np.minimum(self.report_min, np.min(error_history))
        y_max = np.maximum(self.report_max, np.max(error_history))
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_history)
        ax.set_xlabel(f"x{self.reporting_bin_size} iterations")
        ax.set_ylabel("log error")
        ax.set_ylim(y_min, y_max)
        ax.grid()
        fig.savefig(os.path.join(self.report_path, self.report_name))
        plt.close()
