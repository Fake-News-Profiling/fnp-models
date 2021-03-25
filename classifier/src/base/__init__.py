import json
from abc import ABC, abstractmethod
from typing import Union, Any, List

from kerastuner import HyperParameters


class ScopedHyperParameters:
    """ Defines hyper-parameters within a tree of scopes """

    def __init__(self):
        self._scope = {}

    @classmethod
    def from_json(cls, filepath: str) -> "ScopedHyperParameters":
        """ Create a ScopedHyperParameters instance, from a JSON file """
        with open(filepath, "r") as file:
            data = json.load(file)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ScopedHyperParameters":
        """ Create a ScopedHyperParameters instance, from a dict """
        hp = ScopedHyperParameters()
        for k, v in data.items():
            if isinstance(v, dict):
                hp._scope[k] = cls.from_dict(v)
            else:
                hp._scope[k] = v

        return hp

    def get(self, name: Union[str, List[str]]) -> Any:
        """
        Get a value or inner ScopedHyperParameters instance. Takes dot-separated string (or list of strings)
        representing the path to the hyperparameter/scope to return.
        """
        if isinstance(name, list):
            if len(name) == 1:
                return self._scope[name[0]]

            return self.get(name[0]).get(name[1:])
        else:
            names = name.split(".")
            return self.get(names)

    def __getitem__(self, item):
        return self.get(item)


class AbstractModel(ABC):
    """ An abstract base class for evaluating a fake news profiling model """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, x, y):
        """ Fit the underlying model with the given data """
        pass

    @abstractmethod
    def predict(self, x):
        """ Make a prediction with the given data """
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass


def load_hyperparameters(trial_filepath: str) -> HyperParameters:
    """ Load a trials hyper-parameters from a JSON file """
    with open(trial_filepath) as trial_file:
        trial = json.load(trial_file)
        return HyperParameters.from_config(trial["hyperparameters"])
