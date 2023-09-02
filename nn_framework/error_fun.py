import numpy as np


class Sqr:

    @staticmethod
    def calc(x, y):
        return (y - x)**2

    @staticmethod
    def calc_d(x, y):
        return 2 * (y - x)


class Abs:

    @staticmethod
    def calc(x, y):
        return np.abs(y - x)

    @staticmethod
    def calc_d(x, y):
        return np.sign(y - x)
