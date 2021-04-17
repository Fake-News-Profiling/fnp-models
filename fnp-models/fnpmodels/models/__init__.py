import json
from abc import ABC, abstractmethod
from typing import Union, List, Any, Optional


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

    def Fixed(self, name: Union[str, List[str]], value: Any) -> Any:
        """
        Used for compatibility reasons with Keras Tuner's HyperParameters object.
        If the hyperparameters exists, then returns that, else returns `value`.
        """
        try:
            return self.get(name)
        except KeyError:
            return value

    def __getitem__(self, item: Union[str, List[str]]):
        return self.get(item)

    def __contains__(self, item: Union[str, List[str]]):
        try:
            self.get(item)
            return True
        except KeyError:
            return False


class AbstractProcessor(ABC):
    """ Processes some data """

    @abstractmethod
    def __call__(self, x, *args, **kwargs):
        pass

    def transform(self, x, *args, **kwargs):
        self.__call__(x, *args, **kwargs)


class AbstractModel(AbstractProcessor, ABC):
    """ Data processing model """

    def __init__(self, hyperparameters: Optional[ScopedHyperParameters] = None):
        self.hp = hyperparameters

    @abstractmethod
    def train(self, x, y):
        pass
