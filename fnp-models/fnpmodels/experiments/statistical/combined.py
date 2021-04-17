import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from fnpmodels.processing.statistical import readability, ner, sentiment, combined_tweet_extractor
from fnpmodels.experiments import load_hyperparameters
from fnpmodels.experiments.handler import ExperimentHandler
from fnpmodels.experiments.statistical import get_sentiment_wrapper, get_ner_wrapper, AbstractStatisticalExperiment

""" Experiment for training combined and ensemble statistical models """


class VotingClassifier(ClassifierMixin, BaseEstimator):
    """ Voting classifier which votes depending on input data """

    def fit(self, x, y):
        pass

    def predict(self, x):
        # Input is an array of predict_proba outputs (i.e. x.shape == (num_models * num_classes))
        x = x.reshape((len(x), -1, 2))
        return np.argmax(np.sum(x, axis=1), axis=1)

    def predict_proba(self, x):
        x = x.reshape((len(x), -1, 2))
        return np.mean(x, axis=1)


class CombinedStatisticalExperiment(AbstractStatisticalExperiment):
    """ Extract all statistical data at the user-level, and use this to train a single Sklearn model """

    def input_data_transformer(self, x):
        ner_wrapper = get_ner_wrapper(self.hyperparameters)
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)
        extractor = combined_tweet_extractor(ner_wrapper=ner_wrapper, sentiment_wrapper=sentiment_wrapper)
        return extractor.transform(x)


class EnsembleStatisticalExperiment(AbstractStatisticalExperiment):
    """ Load and train statistical models, and then train an Sklearn ensemble model """

    def build_model(self, hp):
        model_type = hp.Choice(
            "Sklearn.model_type",
            [LogisticRegression.__name__, SVC.__name__, RandomForestClassifier.__name__, XGBClassifier.__name__,
             VotingClassifier.__name__]
        )

        if model_type == VotingClassifier.__name__:
            return VotingClassifier()

        return self.select_sklearn_model(hp)

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        def load_fit_model(get_extractor, hyperparameters):
            hp = load_hyperparameters(hyperparameters)
            extractor = get_extractor(hp)
            x_train_model = extractor.transform(x_train)
            x_test_model = extractor.transform(x_test)
            model = super(EnsembleStatisticalExperiment, self).build_model(hp)
            model.fit(x_train_model, y_train)
            return [model.predict_proba(x_train_model), model.predict_proba(x_test_model)]

        models = [
            (lambda hp: readability.readability_tweet_extractor(),
             self.config.hyperparameters["Readability.trial_hp"]),
            (lambda hp: ner.ner_tweet_extractor(get_ner_wrapper(hp)),
             self.config.hyperparameters["Ner.trial_hp"]),
            (lambda hp: sentiment.sentiment_tweet_extractor(get_sentiment_wrapper(hp)),
             self.config.hyperparameters["Sentiment.trial_hp"]),
        ]

        data = np.asarray([load_fit_model(*args) for args in models])
        x_train_out = np.concatenate(data[:, 0], axis=1)
        x_test_out = np.concatenate(data[:, 1], axis=1)
        return x_train_out, y_train, x_test_out, y_test


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            CombinedStatisticalExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "combined_9",
                "max_trials": 100,
                "hyperparameters": {
                    "Ner.library": "spacy",
                    "Ner.spacy_pipeline": "en_core_web_sm",
                    "Sentiment.library": "vader",
                },
            }
        ), (
            EnsembleStatisticalExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "combined_ensemble_9",
                "max_trials": 100,
                "hyperparameters": {
                    "Readability.trial_hp": "../training/statistical/readability_9/"
                                            "trial_17e6e17c62b001acb08029fbd018a480/trial.json",
                    "Ner.trial_hp": "../training/statistical/ner_spacy_sm_9/"
                                    "trial_67bc60cddaffe204c9d613e3e0e5a058/trial.json",
                    "Sentiment.trial_hp": "../training/statistical/sentiment_vader_7/"
                                          "trial_70ab73dc2d9661c28ce343addc2e6420/trial.json",
                },
            }
        )
    ]
    handler = ExperimentHandler(experiments)
    handler.run_experiments(dataset_dir)
    handler.print_results(num_trials=20)
