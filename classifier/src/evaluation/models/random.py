import numpy as np

from evaluation import AbstractEvaluationModel


class RandomModel(AbstractEvaluationModel):
    """ A random model which uses `numpy.random.randint(2)` to pick a prediction value of 0 or 1 """

    def fit(self, x, y):
        pass

    def evaluate(self, x, y, metrics):
        predictions = np.random.randint(2, size=(len(y),)).astype(np.float32)
        return self.evaluate_metrics(metrics, y, predictions)
