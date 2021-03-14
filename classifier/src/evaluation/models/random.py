import numpy as np
from kerastuner import HyperParameters

from base import AbstractModel


class RandomModel(AbstractModel):
    """ A random model which uses `numpy.random.randint()` to pick a prediction value of 0 or 1 """
    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)
        self.name = self.__class__.__name__

    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.random.randint(2, size=(len(x),)).astype(np.float32)
