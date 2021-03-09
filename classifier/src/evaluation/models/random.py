import numpy as np

from evaluation.models import AbstractEvaluationModel


class RandomModel(AbstractEvaluationModel):
    """ A random model which uses `numpy.random.randint()` to pick a prediction value of 0 or 1 """

    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.random.randint(2, size=(len(x),)).astype(np.float32)
