from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from base import AbstractModel, ScopedHyperParameters


class SklearnModel(AbstractModel):
    """ An Sklearn Model """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)
        self.model = None

    def build(self):
        """ Build the Sklearn Pipeline """
        steps = [("StandardScaler", StandardScaler())]

        # Pick the Estimator
        estimator = None
        model_type = self.hyperparameters["model_type"]
        if model_type == "LogisticRegression":
            estimator = LogisticRegression(
                C=self.hyperparameters["C"],
                solver=self.hyperparameters["solver"],
            )
        elif model_type == "SVC":
            estimator = SVC(
                C=self.hyperparameters["C"],
                kernel=self.hyperparameters["kernel"],
                probability=True,
            )
        elif model_type == "RandomForestClassifier":
            estimator = RandomForestClassifier(
                n_estimators=self.hyperparameters["n_estimators"],
                criterion=self.hyperparameters["criterion"],
                min_samples_split=self.hyperparameters["min_samples_split"],
                min_samples_leaf=self.hyperparameters["min_samples_leaf"],
                min_impurity_decrease=self.hyperparameters["min_impurity_decrease"],
            )
        elif model_type == "XGBClassifier":
            estimator = XGBClassifier(
                learning_rate=self.hyperparameters["learning_rate"],
                gamma=self.hyperparameters["gamma"],
                max_depth=self.hyperparameters["max_depth"],
                min_child_weight=self.hyperparameters["min_child_weight"],
                subsample=self.hyperparameters["subsample"],
                colsample_bytree=self.hyperparameters["colsample_bytree"],
                colsample_bylevel=self.hyperparameters["colsample_bylevel"],
                colsample_bynode=self.hyperparameters["colsample_bynode"],
                reg_lambda=self.hyperparameters["reg_lambda"],
                reg_alpha=self.hyperparameters["reg_alpha"],
            )
        steps.append(("Estimator", estimator))

        # Use PCA if not using a Gradient Boosting Classifier (due to multi-collinearity)
        if self.hyperparameters["model_type"] != "XGBClassifier":
            steps.insert(0, ("PCA", PCA()))

        if self.hyperparameters["ignore_preprocessing"]:
            self.model = estimator
        else:
            self.model = Pipeline(steps)

    def fit(self, x, y):
        self.build()
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
