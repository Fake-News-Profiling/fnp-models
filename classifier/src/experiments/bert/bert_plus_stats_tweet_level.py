import sys

import numpy as np
import tensorflow as tf

from bert import BertIndividualTweetTokenizer
from bert.models import shuffle_bert_data, bert_layers
from data import parse_dataset
from experiments.experiment import AbstractBertExperiment
import data.preprocess as pre
from statistical.data_extraction import tweet_level_extractor


tweet_level_stats_extractor = tweet_level_extractor()


def extract_tweet_level_stats(x_train, x_test):
    """ Extract tweet-level statistical data """
    stats_preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
    x_train_stats_pre = stats_preprocessor.transform(x_train)
    x_test_stats_pre = stats_preprocessor.transform(x_test)

    x_train_stats = list(map(tweet_level_stats_extractor.transform, x_train_stats_pre))
    x_test_stats = list(map(tweet_level_stats_extractor.transform, x_test_stats_pre))
    return x_train_stats, x_test_stats


class BertPlusStatsEmbeddingTweetLevelExperiment(AbstractBertExperiment):
    """
    Train a BERT model on individual tweets, where statistical tweet-level data has also been concatenated to each
    tweet
    """

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)

        # Classifier layer
        dense_out = self.single_dense_layer(
            bert_output["pooled_output"],
            dropout_rate=hp.Choice("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0, 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0, 0.0001, 0.001, 0.01]),
            dense_activity_reg=hp.Choice("Bert.dense_activity_reg", [0, 0.0001, 0.001, 0.01]),
        )

        return self.compile_model_with_adamw(bert_input, dense_out)

    @classmethod
    def _preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        # Extract tweet-level statistical features
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test)

        # Preprocess data for BERT
        _, x_train_bert_pre, _, x_test_bert_pre, _ = cls.preprocess_data(hp, x_train, y_train, x_test, y_test)

        # Combine BERT tweet strings with statistical data
        def combine(x_stats, x_bert):
            x_stats_joined = [[" ".join(map(str, tweet)) for tweet in tweet_feed] for tweet_feed in x_stats]
            return np.asarray([
                [
                    x_stats_joined[feed_i][tweet_j] + " | " + x_bert[feed_i][tweet_j]
                    for tweet_j in range(len(x_bert[feed_i]))
                ] for feed_i in range(len(x_bert))])

        x_train_combined = combine(x_train_stats, x_train_bert_pre)
        x_test_combined = combine(x_test_stats, x_test_bert_pre)

        # Tokenize for BERT
        x_train_bert, y_train_bert, x_test_bert, y_test_bert = cls.tokenize_data(
            hp, x_train_combined, y_train, x_test_combined, y_test, tokenizer_class=BertIndividualTweetTokenizer)

        x_train_bert, y_train_bert = shuffle_bert_data(x_train_bert, y_train_bert)
        return hp, x_train_bert, y_train_bert, x_test_bert, y_test_bert


class BertPlusStatsTweetLevelExperiment(AbstractBertExperiment):
    """
    Train a BERT model on individual tweets, where individual tweet statistical data is also inputted into the
    same dense classifier that trains BERT
    """

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)
        bert_input["tweet_level_stats"] = tf.keras.layers.Input(
            (len(tweet_level_stats_extractor.feature_names),), name="input_tweet_level_stats")

        # Concatenate BERT output data with tweet-level statistical data
        pooled_output_and_stats = tf.concat([bert_input["tweet_level_stats"], bert_output["pooled_output"]], -1)

        # Classifier layer
        dense_out = self.single_dense_layer(
            pooled_output_and_stats,
            dropout_rate=hp.Choice("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0, 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0, 0.0001, 0.001, 0.01]),
            dense_activity_reg=hp.Choice("Bert.dense_activity_reg", [0, 0.0001, 0.001, 0.01]),
        )

        return self.compile_model_with_adamw(bert_input, dense_out)

    @classmethod
    def _preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        # Extract tweet-level statistical features
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test)

        # Preprocess data for BERT
        _, x_train_bert_pre, _, x_test_bert_pre, _ = cls.preprocess_data(hp, x_train, y_train, x_test, y_test)

        # Tokenize for BERT
        x_train_bert, y_train_bert, x_test_bert, y_test_bert = cls.tokenize_data(
            hp, x_train_bert_pre, y_train, x_test_bert_pre, y_test, tokenizer_class=BertIndividualTweetTokenizer)
        x_train_bert["tweet_level_stats"] = x_train_stats
        x_test_bert["tweet_level_stats"] = x_test_stats

        x_train_bert, y_train_bert = shuffle_bert_data(x_train_bert, y_train_bert)
        return hp, x_train_bert, y_train_bert, x_test_bert, y_test_bert


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]
    experiment_dir = sys.argv[2]
    x, y = parse_dataset(dataset_dir, "en")

    # BertPlusStatsEmbeddingTweetLevelExperiment
    config = {
        "max_trials": 20,
        "hyperparameters": {
            "epochs": 8,
            "batch_size": 64,
            "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "Bert.hidden_size": 128,
            "Bert.preprocessing": "[remove_emojis, remove_tags]",
        }
    }
    experiment = BertPlusStatsEmbeddingTweetLevelExperiment(experiment_dir, "bert_stats_embeddings_1", config)
    experiment.run(x, y)

    # BertPlusStatsTweetLevelExperiment
    experiment = BertPlusStatsEmbeddingTweetLevelExperiment(experiment_dir, "bert_stats_1", config)
    experiment.run(x, y)
