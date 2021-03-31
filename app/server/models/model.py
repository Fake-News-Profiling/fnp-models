from typing import List, Callable
from abc import ABC, abstractmethod
from functools import reduce


class AbstractDataProcessor(ABC):
    """ Applies functions to transform some data """

    @abstractmethod
    def transform(self, X):
        pass


class DataProcessor(AbstractDataProcessor):
    """ Basic AbstractDataProcessor implementation, given a function """

    def __init__(self, func: Callable):
        self.func = func

    def transform(self, X):
        return self.func(X)


class ModelPipeline(AbstractDataProcessor):
    """
    Combines a list of transformers to form a data pipeline which outputs a model
    prediction
    """

    def __init__(self, name: str, transformers: List[AbstractDataProcessor]):
        self.name = name
        self.transformers = transformers

    def transform(self, X):
        """ Pushes the input_data through the transformer pipeline """
        return reduce(
            lambda data, transformer: transformer.transform(data), self.transformers, X
        )
