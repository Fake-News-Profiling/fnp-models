from typing import List

import numpy as np
from kerastuner import HyperParameters

from base import AbstractModel


class EnsembleModel(AbstractModel):
    """ A random model which uses `numpy.random.randint()` to pick a prediction value of 0 or 1 """
    def __init__(self, hyperparameters: HyperParameters, models: List[AbstractModel]):
        super().__init__(hyperparameters)
        self.models = models
        self.ensemble_model = None

    def fit(self, x, y):
        x_preds = []
        for model in self.models:
            model.fit(x, y)

            x_preds += model.predict(x)  # TODO - fix me

        self.ensemble_model.fit(x_preds, y)

    def predict(self, x):
        return np.random.randint(2, size=(len(x),)).astype(np.float32)
