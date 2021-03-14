import numpy as np

from base import ScopedHyperParameters, AbstractModel
from evaluation.models.sklearn import SklearnModel


class EnsembleModel(AbstractModel):
    """ Combines multiple model predictions into one user classification """

    def __init__(self, hyperparameters: ScopedHyperParameters, models):
        super().__init__(hyperparameters)
        self.models = [model(hyperparameters.get_scope(scope)) for model, scope in models]
        self.sklearn_model = SklearnModel(hyperparameters.get_scope("SklearnModel"))

    def _get_predict_probas(self, x):
        xt = [model.predict_proba(x) for model in self.models]
        return np.concatenate(xt, axis=-1)

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)

        xt = self._get_predict_probas(x)
        self.sklearn_model.fit(xt, y)

    def predict(self, x):
        xt = self._get_predict_probas(x)
        return self.sklearn_model.predict(xt)

    def predict_proba(self, x):
        xt = self._get_predict_probas(x)
        return self.sklearn_model.predict_proba(xt)
