import sys

import numpy as np
import tensorflow as tf

import data.preprocess as pre
from experiments.bert.downstream_loss.bert_experiment_models import BertTrainedOnDownstreamLoss, BertUserLevelClassifier
from experiments.experiment import ExperimentConfig
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel
from statistical.data_extraction import tweet_level_extractor
from statistical.data_extraction.sentiment.vader import VaderSentimentAnalysisWrapper


TWEET_FEED_LEN = 10
tweet_level_stats_extractor = tweet_level_extractor(VaderSentimentAnalysisWrapper())


def extract_tweet_level_stats(x_train, x_test):
    """ Extract tweet-level statistical data """
    stats_preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
    x_train_stats_pre = stats_preprocessor.transform(x_train)
    x_test_stats_pre = stats_preprocessor.transform(x_test)

    x_train_stats = tf.convert_to_tensor(list(map(tweet_level_stats_extractor.transform, x_train_stats_pre)))
    x_test_stats = tf.convert_to_tensor(list(map(tweet_level_stats_extractor.transform, x_test_stats_pre)))
    return x_train_stats, x_test_stats


class BertPlusStatsUserLevelClassifier(BertUserLevelClassifier):
    def call(self, inputs, training=None, mask=None):
        # inputs.shape == [(batch_size, TWEET_FEED_LEN, 3, Bert.hidden_size),
        #                  (batch_size, TWEET_FEED_LEN, NUM_STATS_FEATURES)]
        # Returns a tensor with shape (batch_size, 1)
        bert_data = inputs[0]
        stats_data = inputs[1]
        x_train = self._accumulate_bert_outputs(bert_data, training=training)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        x_train = tf.concat([stats_data, x_train], axis=-1)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size + NUM_STATS_FEATURES)
        x_train = self.pooling(x_train)
        # x_train.shape == (batch_size, Bert.hidden_size + NUM_STATS_FEATURES)
        x_train = self.dropout(x_train, training=training)
        return x_train


class BertPlusStatsExperiment(BertTrainedOnDownstreamLoss):
    """
    Train a BERT model on individual tweets, where downstream user loss is used to train BERT. Individual tweet stats
    are concatenated with BERT pooled_output to train a final linear dense classifier
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.tuner.num_folds = 3

    def build_model(self, hp):
        bert_input = tf.keras.layers.Input(
            (TWEET_FEED_LEN, 3, hp.get("Bert.hidden_size")), dtype=tf.int32, name="BERT_input")
        stats_input = tf.keras.layers.Input(
            (TWEET_FEED_LEN, len(tweet_level_stats_extractor.feature_names)), dtype=tf.float32, name="stats_input")
        inputs = [bert_input, stats_input]

        bert_outputs = BertPlusStatsUserLevelClassifier(hp)(inputs)
        linear = tf.keras.layers.Dense(
            1,
            activation=hp.Fixed("Bert.dense_activation", "linear"),
            kernel_regularizer=tf.keras.regularizers.l2(hp.get("Bert.dense_kernel_reg")),
        )(bert_outputs)
        return CompileOnFitKerasModel(inputs, linear, optimizer_learning_rate=hp.get("learning_rate"))

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        _, x_train_bert, y_train, x_test_bert, y_test = super(BertPlusStatsExperiment, self).preprocess_cv_data(
            self.hyperparameters, x_train, y_train, x_test, y_test, shuffle_data=False)
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test)

        def process_stats(x):
            x = tf.cast(x, dtype=tf.float32)
            return tf.reshape(x, shape=(-1, TWEET_FEED_LEN, len(tweet_level_stats_extractor.feature_names)))

        x_train_stats = process_stats(x_train_stats)
        x_test_stats = process_stats(x_test_stats)

        # Shuffle training data
        shuffle_seed = 1
        tf.random.set_seed(shuffle_seed)
        x_train_stats = tf.random.shuffle(x_train_stats, seed=shuffle_seed)
        tf.random.set_seed(shuffle_seed)
        x_train_bert = tf.random.shuffle(x_train_bert, seed=shuffle_seed)
        tf.random.set_seed(shuffle_seed)
        y_train = tf.random.shuffle(y_train, seed=shuffle_seed)

        return [x_train_bert, x_train_stats], y_train, [x_test_bert, x_test_stats], y_test

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return hp, x_train, y_train, x_test, y_test


class BertPlusStatsEmbeddingExperiment(BertTrainedOnDownstreamLoss):
    """
    Train a BERT model on individual tweets, where downstream user loss is used to train BERT. Individual tweets are
    concatenated with statistical features, so inputted tweets are of the form:
    "<stat_1> <stat_2> ... <stat_n> | <tweet>"
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.tuner.num_folds = 3

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
            BertPlusStatsEmbeddingExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss_plus_stats_embedding",
                "experiment_name": "indiv_1",
                "max_trials": 2,
                "hyperparameters": {
                    "epochs": 10,
                    "batch_size": [8, 8],
                    "learning_rate": 2e-5,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags]",
                    "Bert.pooler": ["max", "concat"],
                    "selected_encoder_outputs": "sum_last_4_hidden_layers",
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_kernel_reg": 0,
                },
            }
        ), (
            BertPlusStatsExperiment,
            {
                "experiment_dir": "/content/training/bert_clf/downstream_loss_plus_stats",
                "experiment_name": "indiv_1",
                "max_trials": 2,
                "hyperparameters": {
                    "epochs": 10,
                    "batch_size": [8, 8],
                    "learning_rate": 2e-5,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags]",
                    "Bert.pooler": ["max", "concat"],
                    "selected_encoder_outputs": "sum_last_4_hidden_layers",
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_kernel_reg": 0,
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        handler.run_experiments(dataset_dir)
