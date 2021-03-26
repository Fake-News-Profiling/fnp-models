import sys

import numpy as np
import tensorflow as tf

import data.preprocess as pre
from experiments.bert.downstream_loss.bert_downstream_loss import BertTrainedOnDownstreamLoss
from experiments.handler import ExperimentHandler
from statistical.data_extraction import tweet_level_extractor
from statistical.data_extraction.sentiment import VaderSentimentAnalysisWrapper

tweet_level_stats_extractor = tweet_level_extractor(VaderSentimentAnalysisWrapper())


def extract_tweet_level_stats(x_train, x_test):
    """ Extract tweet-level statistical data """
    stats_preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
    x_train_stats_pre = stats_preprocessor.transform(x_train)
    x_test_stats_pre = stats_preprocessor.transform(x_test)

    x_train_stats = list(map(tweet_level_stats_extractor.transform, x_train_stats_pre))
    x_test_stats = list(map(tweet_level_stats_extractor.transform, x_test_stats_pre))
    return x_train_stats, x_test_stats


class BertPlusStatsEmbeddingExperiment(BertTrainedOnDownstreamLoss):
    """
    Train a BERT model on individual tweets, where statistical tweet-level data has also been concatenated to each
    tweet such that each input datapoint is of the form: "<stat_1> <stat_2> ... <stat_n> | <tweet>"
    """

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        # Extract tweet-level statistical features
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test)

        # Pre-process tweet data for BERT
        _, x_train_pre, _, x_test_pre, _ = self.preprocess_data(
            self.hyperparameters, x_train, y_train, x_test, y_test)

        # Combine BERT tweet strings with statistical data
        def combine(x_stats, x_bert):
            x_stats_joined = [[" ".join(map(str, tweet)) for tweet in tweet_feed] for tweet_feed in x_stats]
            return np.asarray([
                [
                    x_stats_joined[feed_i][tweet_j] + " | " + x_bert[feed_i][tweet_j]
                    for tweet_j in range(len(x_bert[feed_i]))
                ] for feed_i in range(len(x_bert))])

        x_train_combined = combine(x_train_stats, x_train_pre)
        x_test_combined = combine(x_test_stats, x_test_pre)

        # Tokenize for BERT (tokenizing individual user tweets)
        return self.tokenize_cv_data(self.hyperparameters, x_train_combined, y_train, x_test_combined, y_test)[1:]

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return hp, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            # Bert (128) Individual with different pooled output methods
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "indiv_1",
                "max_trials": 1,
                "hyperparameters": {
                    "epochs": 16,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags, remove_punctuation]",
                    "selected_encoder_outputs": "default",
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        handler.run_experiments(dataset_dir)
