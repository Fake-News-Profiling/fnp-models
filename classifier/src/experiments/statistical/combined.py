import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier

from base import load_hyperparameters
from experiments.experiment import AbstractSklearnExperiment, ExperimentConfig
import statistical.data_extraction as ex
from experiments.statistical import get_sentiment_wrapper, get_ner_wrapper

""" Experiment for training combined and ensemble statistical models """


class CombinedStatisticalExperiment(AbstractSklearnExperiment):
    """ Extract all statistical data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        ner_wrapper = get_ner_wrapper(self.hyperparameters)
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)
        extractor = ex.combined_tweet_extractor(ner_wrapper=ner_wrapper, sentiment_wrapper=sentiment_wrapper)
        return extractor.transform(x)


class EnsembleStatisticalExperiment(AbstractSklearnExperiment):
    """ Load and train statistical models, and then train an Sklearn ensemble model """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.num_readability_features = self.num_ner_features = self.num_sentiment_features = 0

    def build_model(self, hp):
        readability_hp = load_hyperparameters(hp.get("Readability.trial_hp"))
        readability_model = Pipeline(
            [("data_ex", self.extract_data_transformer(np.s_[:self.num_readability_features]))] +
            super().build_model(readability_hp).steps
        )
        ner_hp = load_hyperparameters(hp.get("Ner.trial_hp"))
        ner_model = Pipeline(
            [("data_ex", self.extract_data_transformer(np.s_[self.num_readability_features:self.num_ner_features]))] +
            super().build_model(ner_hp).steps
        )
        sentiment_hp = load_hyperparameters(hp.get("Sentiment.trial_hp"))
        sentiment_model = Pipeline(
            [("data_ex", self.extract_data_transformer(np.s_[self.num_ner_features:self.num_sentiment_features]))] +
            super().build_model(sentiment_hp).steps
        )

        model_type = hp.Choice(
            "Sklearn.model_type",
            [LogisticRegression.__name__, SVC.__name__, RandomForestClassifier.__name__, XGBClassifier.__name__,
             VotingClassifier.__name__]
        )

        if model_type == VotingClassifier.__name__:
            return VotingClassifier(
                estimators=[("Readability", readability_model), ("Ner", ner_model), ("Sentiment", sentiment_model)],
                voting=hp.Choice("Sklearn.voting", ["soft", "hard"]),
            )

        return self.select_sklearn_model(hp)

    def input_data_transformer(self, x):
        ner_wrapper = get_ner_wrapper(self.hyperparameters)
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)

        readability_extractor = ex.readability_tweet_extractor()
        self.num_readability_features = len(readability_extractor.feature_names)
        ner_extractor = ex.ner_tweet_extractor(ner_wrapper)
        self.num_ner_features = len(ner_extractor.feature_names)
        sentiment_extractor = ex.sentiment_tweet_extractor(sentiment_wrapper)
        self.num_sentiment_features = len(sentiment_extractor.feature_names)

        extractor = ex.TweetStatsExtractor([
            *readability_extractor.extractors,
            *ner_extractor.extractors,
            *sentiment_extractor.extractors,
        ])
        extractor.feature_names = [
            *readability_extractor.feature_names,
            *ner_extractor.feature_names,
            *sentiment_extractor.feature_names,
        ]
        return extractor.transform(x)

    @staticmethod
    def extract_data_transformer(x_slice):
        def extract(x):
            return np.asarray(x)[:, x_slice]
        return FunctionTransformer(extract)
