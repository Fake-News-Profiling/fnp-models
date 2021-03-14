import json
from abc import ABC, abstractmethod

from kerastuner import HyperParameters


class ScopedHyperParameters:
    """ Defines hyper-parameters within a tree of scopes """
    def __init__(self):
        super().__init__()
        self.scope = {}
        self.value = None

    @staticmethod
    def _add_to_scope(hyperparameters, scope_names, value):
        if len(scope_names) == 0:
            # Set the hyperparameters value
            hyperparameters.value = value
        else:
            # Go in to the next scope
            next_scope_name = scope_names[0]
            if next_scope_name not in hyperparameters.scope:
                hyperparameters.scope[next_scope_name] = ScopedHyperParameters()

            ScopedHyperParameters._add_to_scope(hyperparameters.scope[next_scope_name], scope_names[1:], value)

    @staticmethod
    def _get_from_scope(hyperparameters, scope_names):
        if len(scope_names) == 0:
            # Get the hyperparameters value
            return hyperparameters.value
        else:
            # Go in to the next scope
            return ScopedHyperParameters._get_from_scope(hyperparameters.scope[scope_names[0]], scope_names[1:])

    @staticmethod
    def from_hyperparameters(hyperparameters: HyperParameters, scoped_hyperparameters=None):
        """ Convert Keras Tuner HyperParameters to a ScopedHyperParameters object """
        if scoped_hyperparameters is None:
            scoped_hyperparameters = ScopedHyperParameters()
        for k, v in hyperparameters.values.items():
            ScopedHyperParameters._add_to_scope(scoped_hyperparameters, str(k).split("."), v)

        return scoped_hyperparameters

    def add_hyperparameters(self, hyperparameters: HyperParameters):
        """ Add hyper-parameters from a HyperParameters object to this ScopedHyperParameters object """
        self.from_hyperparameters(hyperparameters, scoped_hyperparameters=self)

    def get_scope(self, scope_name):
        """ Return ScopedHyperParameters for the given scope name """
        return self.scope[scope_name]

    def get(self, hyperparameter_name: str):
        """ Get a value for a hyper-parameter """
        return self._get_from_scope(self, hyperparameter_name.split("."))


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


def load_hyperparameters(trial_filepath, to_scoped_hyperparameters=False):
    """ Load a trials hyper-parameters from a JSON file """
    with open(trial_filepath) as trial_file:
        trial = json.load(trial_file)
        hyperparameters = HyperParameters.from_config(trial["hyperparameters"])
        if to_scoped_hyperparameters:
            return ScopedHyperParameters.from_hyperparameters(hyperparameters)

        return hyperparameters
