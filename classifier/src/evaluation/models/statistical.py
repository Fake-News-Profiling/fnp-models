import numpy as np
from kerastuner import HyperParameters
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from base import AbstractModel
import data.preprocess as pre
from statistical.data_extraction import readability_tweet_extractor, ner_tweet_extractor, sentiment_tweet_extractor


class StatisticalModel(AbstractModel):
    """
    User-level statistical fake news profiling model, which uses readability, named-entity recognition, and
    sentiment (or all 3) features to make predictions
    """

    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)
        self.name = self.hyperparameters.get("StatisticalModel_name")

        # Pick underlying model
        model_type = self.hyperparameters.get("StatisticalModel_model_type")
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(
                C=self.hyperparameters.get("StatisticalModel_C"),
                solver=self.hyperparameters.get("StatisticalModel_solver"),
            )
        elif model_type == "SVC":
            self.model = SVC(
                C=self.hyperparameters.get("StatisticalModel_C"),
                kernel=self.hyperparameters.get("StatisticalModel_kernel"),
                probability=True,
            )
        elif model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(
                n_estimators=self.hyperparameters.get("StatisticalModel_n_estimators"),
                criterion=self.hyperparameters.get("StatisticalModel_criterion"),
                min_samples_split=self.hyperparameters.get("StatisticalModel_min_samples_split"),
                min_samples_leaf=self.hyperparameters.get("StatisticalModel_min_samples_leaf"),
                min_impurity_decrease=self.hyperparameters.get("StatisticalModel_min_impurity_decrease"),
            )
        elif model_type == "XGBClassifier":
            self.model = XGBClassifier(
                learning_rate=self.hyperparameters.get("StatisticalModel_learning_rate"),
                gamma=self.hyperparameters.get("StatisticalModel_gamma"),
                max_depth=self.hyperparameters.get("StatisticalModel_max_depth"),
                min_child_weight=self.hyperparameters.get("StatisticalModel_min_child_weight"),
                subsample=self.hyperparameters.get("StatisticalModel_subsample"),
                colsample_bytree=self.hyperparameters.get("StatisticalModel_colsample_bytree"),
                colsample_bylevel=self.hyperparameters.get("StatisticalModel_colsample_bylevel"),
                colsample_bynode=self.hyperparameters.get("StatisticalModel_colsample_bynode"),
                reg_lambda=self.hyperparameters.get("StatisticalModel_reg_lambda"),
                reg_alpha=self.hyperparameters.get("StatisticalModel_reg_alpha"),
            )
        else:
            raise RuntimeError("Invalid model type")

        # Preprocessing and data extraction
        data_type = self.hyperparameters.get("StatisticalModel_data_type")
        if data_type == "readability":
            self.extractors = [readability_tweet_extractor()]
        elif data_type == "ner":
            self.extractors = [ner_tweet_extractor()]
        elif data_type == "sentiment":
            self.extractors = [sentiment_tweet_extractor()]
        elif data_type == "combined":
            self.extractors = [readability_tweet_extractor(), ner_tweet_extractor(), sentiment_tweet_extractor()]
        else:
            raise RuntimeError("Invalid data type extractor")
        self.pca = PCA()
        self.scaler = StandardScaler()
        self.preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])

    def _extract(self, x):
        x_processed = self.preprocessor.transform(x)
        x_extracted = [ex.transform(x_processed) for ex in self.extractors]
        return np.concatenate(x_extracted, axis=1) if len(x_extracted) > 1 else x_extracted[0]

    def fit(self, x, y):
        x_extracted = self._extract(x)
        x_pca = self.pca.fit_transform(x_extracted, y)
        x_scaled = self.scaler.fit_transform(x_pca, y)
        self.model.fit(x_scaled, y)

    def predict(self, x):
        x_extracted = self._extract(x)
        x_pca = self.pca.transform(x_extracted)
        x_scaled = self.scaler.transform(x_pca)
        return self.model.predict(x_scaled)

