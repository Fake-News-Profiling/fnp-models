from abc import ABC, abstractmethod
from typing import Collection

from kerastuner import HyperParameters

from evaluation.models.bert_model import BertPooledModel
from evaluation.models.buda20 import Buda20NgramEnsembleModel
from evaluation.models.random import RandomModel


class AbstractModel(ABC):
    """ An abstract base class for evaluating a fake news profiling model """
    def __init__(self, hyperparameters: HyperParameters):
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, x: Collection[Collection[any]], y: Collection[float]):
        """ Fit the underlying model with the given data """
        pass

    @abstractmethod
    def predict(self, x: Collection[Collection[any]]) -> Collection[float]:
        """ Make a prediction with the given data """
        pass
