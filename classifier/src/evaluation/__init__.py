from abc import ABC, abstractmethod


class AbstractEvaluationModel(ABC):
    """ An abstract base class for evaluating a fake news profiling model """

    @staticmethod
    def evaluate_metrics(metrics, y_true, y_pred):
        return {name: fn(y_true, y_pred) for name, fn in metrics}

    @abstractmethod
    def fit(self, x, y):
        """ Fit the underlying model with the given data """
        pass

    @abstractmethod
    def evaluate(self, x, y, metrics):
        """ Evaluate the performance of the underlying model, returning a dict of metric results """
        pass
