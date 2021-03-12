import json
from abc import ABC, abstractmethod

from kerastuner import HyperParameters


class AbstractModel(ABC):
    """ An abstract base class for evaluating a fake news profiling model """
    def __init__(self, hyperparameters: HyperParameters):
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, x, y):
        """ Fit the underlying model with the given data """
        pass

    @abstractmethod
    def predict(self, x):
        """ Make a prediction with the given data """
        pass


def load_hyperparameters(trial_filepath):
    """ Load a trials hyperparameters from a JSON file """
    with open(trial_filepath) as trial_file:
        trial = json.load(trial_file)
        return HyperParameters.from_config(trial["hyperparameters"])
