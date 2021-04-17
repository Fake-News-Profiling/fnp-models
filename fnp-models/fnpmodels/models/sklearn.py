import numpy as np
from joblib import dump, load
from sklearn.base import ClassifierMixin, BaseEstimator

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import ScopedHyperParameters, AbstractModel


class SklearnModel(AbstractModel):
    """ An Sklearn Model """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)

        if "weights_path" in self.hp:
            self.model = load(self.hp["weights_path"] + ".joblib")
        else:
            self.model = None
            self._build()

    def _build(self):
        """ Build the Sklearn Pipeline """
        steps = [("StandardScaler", StandardScaler())]

        # Pick the Estimator
        estimator = None
        model_type = self.hp["model_type"]
        if model_type == "LogisticRegression":
            estimator = LogisticRegression(
                C=self.hp["C"],
            )
        elif model_type == "SVC":
            estimator = SVC(
                C=self.hp["C"],
                probability=True,
            )
        elif model_type == "RandomForestClassifier":
            estimator = RandomForestClassifier(
                n_estimators=self.hp["n_estimators"],
                criterion=self.hp["criterion"],
                min_samples_split=self.hp["min_samples_split"],
                min_samples_leaf=self.hp["min_samples_leaf"],
                min_impurity_decrease=self.hp["min_impurity_decrease"],
            )
        elif model_type == "XGBClassifier":
            estimator = XGBClassifier(
                learning_rate=self.hp["learning_rate"],
                gamma=self.hp["gamma"],
                max_depth=self.hp["max_depth"],
                min_child_weight=self.hp["min_child_weight"],
                subsample=self.hp["subsample"],
                colsample_bytree=self.hp["colsample_bytree"],
                colsample_bylevel=self.hp["colsample_bylevel"],
                colsample_bynode=self.hp["colsample_bynode"],
                reg_lambda=self.hp["reg_lambda"],
            )
        elif model_type == "VotingClassifier":
            self.estimator = VotingClassifier
            return
        else:
            raise RuntimeError("Invalid SkLearn model type:", model_type)

        steps.append(("Estimator", estimator))

        # Use PCA if not using a Gradient Boosting Classifier (due to multi-collinearity)
        if self.hp["model_type"] != "XGBClassifier":
            steps.insert(0, ("PCA", PCA()))

        if self.hp["ignore_preprocessing"]:
            self.model = estimator
        else:
            self.model = Pipeline(steps)

    def train(self, x, y):
        self.model.fit(x, y)

        if "weights_path" in self.hp:
            dump(self.model, self.hp["weights_path"] + ".joblib")

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def __call__(self, x, *args, **kwargs):
        return self.predict_proba(x)


class VotingClassifier(ClassifierMixin, BaseEstimator):
    """ Voting classifier which votes depending on input data """

    def fit(self, x, y):
        pass

    def predict(self, x):
        x = self.to_probas(x)
        return np.argmax(np.sum(x, axis=1), axis=1).astype(np.float64)

    def predict_proba(self, x):
        x = self.to_probas(x)
        return np.mean(x, axis=1)

    @staticmethod
    def to_probas(x):
        # x.shape == (-1, TWEET_FEED_LEN)
        x = x.reshape(len(x), -1, 1)
        x_other = 1 - x
        x_probas = np.concatenate([x_other, x], axis=-1)
        # x_probas.shape == (-1, TWEET_FEED_LEN, 2)
        return x_probas