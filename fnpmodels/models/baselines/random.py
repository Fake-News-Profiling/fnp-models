import numpy as np

from fnpmodels.models import AbstractModel


class RandomModel(AbstractModel):
    """ A random model which uses `numpy.random.randint()` to pick a prediction value of 0 or 1 """

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def train(self, x, y):
        pass

    def predict(self, x):
        return np.random.randint(2, size=(len(x),)).astype(np.float32)

    def predict_proba(self, x):
        pass

    def __call__(self, x, *args, **kwargs):
        return self.predict(x)
