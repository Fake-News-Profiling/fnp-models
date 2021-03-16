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
        model_type = self.hyperparameters.get("model_type")
        if model_type == "LogisticRegression":
            estimator = LogisticRegression(
                C=self.hyperparameters.get("C"),
                solver=self.hyperparameters.get("solver"),
            )
        elif model_type == "SVC":
            estimator = SVC(
                C=self.hyperparameters.get("C"),
                kernel=self.hyperparameters.get("kernel"),
                probability=True,
            )
        elif model_type == "RandomForestClassifier":
            estimator = RandomForestClassifier(
                n_estimators=self.hyperparameters.get("n_estimators"),
                criterion=self.hyperparameters.get("criterion"),
                min_samples_split=self.hyperparameters.get("min_samples_split"),
                min_samples_leaf=self.hyperparameters.get("min_samples_leaf"),
                min_impurity_decrease=self.hyperparameters.get("min_impurity_decrease"),
            )
        elif model_type == "XGBClassifier":
            estimator = XGBClassifier(
                learning_rate=self.hyperparameters.get("learning_rate"),
                gamma=self.hyperparameters.get("gamma"),
                max_depth=self.hyperparameters.get("max_depth"),
                min_child_weight=self.hyperparameters.get("min_child_weight"),
                subsample=self.hyperparameters.get("subsample"),
                colsample_bytree=self.hyperparameters.get("colsample_bytree"),
                colsample_bylevel=self.hyperparameters.get("colsample_bylevel"),
                colsample_bynode=self.hyperparameters.get("colsample_bynode"),
                reg_lambda=self.hyperparameters.get("reg_lambda"),
                reg_alpha=self.hyperparameters.get("reg_alpha"),
            )
        steps.append(("Estimator", estimator))

        # Use PCA if not using a Gradient Boosting Classifier (due to multi-collinearity)
        if self.hyperparameters.get("model_type") != "XGBClassifier":
            steps.insert(0, ("PCA", PCA()))

        if self.hyperparameters.get("ignore_preprocessing"):
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
