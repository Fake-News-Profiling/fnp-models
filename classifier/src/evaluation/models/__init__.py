from abc import ABC, abstractmethod

from evaluation.models.random import RandomModel
from evaluation.models.buda20 import Buda20NgramEnsembleModel


class AbstractEvaluationModel(ABC):
    """ An abstract base class for evaluating a fake news profiling model """

    @abstractmethod
    def fit(self, x, y):
        """ Fit the underlying model with the given data """
        pass

    @abstractmethod
    def predict(self, x):
        """ Make a prediction with the given data """
        pass

    def evaluate(self, x, y, metrics):
        """ Evaluate the performance of the underlying model, returning a dict of metric results """
        predictions = self.predict(x)
        return {name: fn(y, predictions) for name, fn in metrics}
