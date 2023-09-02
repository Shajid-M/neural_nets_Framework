import numpy as np


class Tanh:

    @staticmethod
    def calc(data_in):
        data_out = np.tanh(data_in)
        return data_out

    @staticmethod
    def calc_d(data_in):
        v = Tanh.calc(data_in)
        data_out = 1 - (Tanh.calc(v) ** 2)
        return data_out


class Logistic:

    @staticmethod
    def calc(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def calc_d(v):
        return Logistic.calc(v) * (1 - Logistic.calc(v))


class Relu:

    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = 0
        if v>0:
            derivative = 1
        return derivative
