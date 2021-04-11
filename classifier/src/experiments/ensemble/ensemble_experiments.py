import sys
from typing import Dict

import numpy as np
import tensorflow as tf

import statistical.data_extraction as stats
from data.preprocess import BertTweetPreprocessor, tag_indicators, replace_xml_and_html
from experiments.bert.bert_combined import (
    BertDownstreamLossLogitsCombinedExperiment,
    BertDownstreamLossPooledCombinedExperiment,
)
from experiments.experiment import AbstractSklearnExperiment, ExperimentConfig
from experiments.handler import ExperimentHandler
from experiments.statistical import get_sentiment_wrapper, get_ner_wrapper


class AbstractEnsembleExperiment(AbstractSklearnExperiment):
    """
    Fit a final ensemble voting model which combines BERT outputs with user-level statistical features.
    """

    def __init__(self, config: ExperimentConfig, child_model_hyperparameters: Dict[str, Dict]):
        super().__init__(config)
        self.child_hyperparameters = {k: self.parse_to_hyperparameters(hp)
                                      for k, hp in child_model_hyperparameters.items()}

    def input_data_transformer(self, x):
        # x.shape = (num_users, 100)
        tweet_preprocessor = BertTweetPreprocessor([tag_indicators, replace_xml_and_html])
        x_preprocessed = tweet_preprocessor.transform(x)

        # Readability stats
        read_extractor = stats.readability_tweet_extractor()
        x_read = read_extractor.transform(x_preprocessed)

        # Sentiment stats
        sent_extractor = stats.sentiment_tweet_extractor(get_sentiment_wrapper(self.child_hyperparameters["sentiment"]))
        x_sent = sent_extractor.transform(x_preprocessed)

        # NER stats
        ner_extractor = stats.ner_tweet_extractor(get_ner_wrapper(self.child_hyperparameters["ner"]))
        x_ner = ner_extractor.transform(x_preprocessed)

        return np.asarray([[x[i], x_read[i], x_sent[i], x_ner[i]] for i in range(len(x))])

    @staticmethod
    def extract_data(x, x_index):
        return np.asarray([tweet_feed[x_index] for tweet_feed in x])


class EnsembleVotingExperiment(AbstractEnsembleExperiment):
    """
    Fit a final ensemble voting model which votes based on probabilities from BERT user-level classifier and 
    statistical model classifiers.
    """

    def build_model(self, hp):
        return BertDownstreamLossLogitsCombinedExperiment.build_model(hp)

    def cv_data_transformer(self, hp, x_train, y_train, x_test, y_test):
        # Fit BERT
        bert_hp = self.child_hyperparameters["bert"]
        x_train_bert = self.extract_data(x_train, 0)
        x_test_bert = self.extract_data(x_test, 0)

        bert_experiment = BertDownstreamLossLogitsCombinedExperiment if bert_hp.get("Bert.pooling_type") == "logits" \
            else BertDownstreamLossPooledCombinedExperiment
        x_train_bert, _, x_test_bert, _ = bert_experiment.cv_data_transformer(
            bert_hp, x_train_bert, y_train, x_test_bert, y_test)

        bert_model = bert_experiment.build_model(bert_hp)
        bert_model.fit(x_train_bert, y_train)
        x_train_bert = bert_model.predict_proba(x_train_bert)
        x_test_bert = bert_model.predict_proba(x_test_bert)

        # Fit statistical models
        def train_fit_stats_model(hp, x_index):
            x_train_stats = self.extract_data(x_train, x_index)
            x_test_stats = self.extract_data(x_test, x_index)
            stats_model = AbstractSklearnExperiment.build_model(hp)
            stats_model.fit(x_train_stats, y_train)
            return stats_model.predict_proba(x_train_stats), stats_model.predict_proba(x_test_stats)

        x_train_read, x_test_read = train_fit_stats_model(self.child_hyperparameters["readability"], 1)
        x_train_sent, x_test_sent = train_fit_stats_model(self.child_hyperparameters["sentiment"], 2)
        x_train_ner, x_test_ner = train_fit_stats_model(self.child_hyperparameters["ner"], 3)

        # Concatenate predicted probabilities
        x_train_proba = np.concatenate([x_train_bert, x_train_read, x_train_sent, x_train_ner], axis=1)
        x_test_proba = np.concatenate([x_test_bert, x_test_read, x_test_sent, x_test_ner], axis=1)
        return x_train_proba, y_train, x_test_proba, y_test


class EnsemblePoolingExperiment(AbstractEnsembleExperiment):
    """ Fit a final ensemble voting model which pools BERT outputs with user-level statistical features """

    def cv_data_transformer(self, hp, x_train, y_train, x_test, y_test):
        # Fit BERT and extract outputs
        bert_hp = self.child_hyperparameters["bert"]
        x_train_bert = self.extract_data(x_train, 0)
        x_test_bert = self.extract_data(x_test, 0)

        bert_experiment = BertDownstreamLossLogitsCombinedExperiment if bert_hp.get("Bert.pooling_type") == "logits" \
            else BertDownstreamLossPooledCombinedExperiment
        x_train_bert, _, x_test_bert, _ = bert_experiment.cv_data_transformer(
            bert_hp, x_train_bert, y_train, x_test_bert, y_test)

        # Combine statistical data
        x_train_stats = np.asarray([np.concatenate(tweet_feed[1:]) for tweet_feed in x_train])
        x_test_stats = np.asarray([np.concatenate(tweet_feed[1:]) for tweet_feed in x_test])

        # Concatenate predicted probabilities
        x_train_combined = np.concatenate([x_train_bert, x_train_stats], axis=1)
        x_test_combined = np.concatenate([x_test_bert, x_test_stats], axis=1)
        return x_train_combined, y_train, x_test_combined, y_test


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            # Ensemble voting
            EnsembleVotingExperiment,
            {
                "experiment_dir": "../training/bert_clf/ensemble",
                "experiment_name": "voting",
                "max_trials": 100,
                "num_cv_splits": 5,
                "hyperparameters": {
                },
                "hp_filepaths": {
                    "bert": "",
                    "readability": "",
                    "sentiment": "",
                    "ner": "",
                },
            }
        ), (
            # Ensemble voting
            EnsemblePoolingExperiment,
            {
                "experiment_dir": "../training/bert_clf/ensemble",
                "experiment_name": "pooling",
                "max_trials": 100,
                "num_cv_splits": 5,
                "hyperparameters": {
                },
                "hp_filepaths": {
                    "bert": "",
                    "readability": "",
                    "sentiment": {"Sentiment.library": "vader"},
                    "ner": {"Ner.library": "vader"},
                },
            }
        )
    ]

    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        # handler.run_experiments(dataset_dir)
        handler.print_results(10)
