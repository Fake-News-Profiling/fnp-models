import sys
from functools import reduce

import numpy as np
import tensorflow as tf

from bert import BertIndividualTweetTokenizer
from bert.models import shuffle_bert_data, bert_tokenizer
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
    tweet such that each input datapoint is of the form: "<stat_1> <stat_2> ... <stat_n> | <tweet>"
    """

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)

        # Classifier layer
        dense_out = self.single_dense_layer(
            bert_output["pooled_output"],
            dropout_rate=hp.Float("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0., 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0., 0.0001, 0.001, 0.01]),
            dense_activity_reg=hp.Choice("Bert.dense_activity_reg", [0., 0.0001, 0.001, 0.01]),
        )

        return self.compile_model_with_adamw(hp, bert_input, dense_out)

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        # Extract tweet-level statistical features
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test)

        # Preprocess data for BERT
        _, x_train_bert_pre, _, x_test_bert_pre, _ = self.preprocess_data(
            self.hyperparameters, x_train, y_train, x_test, y_test)

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

        # Tokenize for BERT (tokenizing individual user tweets)
        tokenizer = bert_tokenizer(
            self.hyperparameters.get("Bert.encoder_url"),
            self.hyperparameters.get("Bert.hidden_size"),
            BertIndividualTweetTokenizer
        )

        def tokenize(x_combined):
            return [tokenizer.tokenize_input([tweet_feed]) for tweet_feed in x_combined]

        # type(x_train_bert) == List[List[Dict[str, tf.Tensor]]]
        x_train_bert = tokenize(x_train_combined)
        x_test_bert = tokenize(x_test_combined)
        return x_train_bert, y_train, x_test_bert, y_test

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        # Flatten fitted cv user lists: List[Dict[str, tf.Tensor]] -> Dict[str, tf.Tensor]
        tweets_per_user = len(x_train[0]["input_word_ids"])

        def flatten_dicts(x_cv):
            def merge_dicts(d1, d2):
                for k, v in d1.items():
                    d1[k] = tf.concat([v, d2[k]], axis=0)

                return d1
            return reduce(merge_dicts, x_cv)

        x_train_flat = flatten_dicts(x_train)
        x_test_flat = flatten_dicts(x_test)
        y_train_flat = np.asarray([v for i, v in enumerate(y_train) for _ in range(tweets_per_user)])
        y_test_flat = np.asarray([v for i, v in enumerate(y_test) for _ in range(tweets_per_user)])

        assert len(x_train_flat["input_word_ids"]) == len(y_train_flat)
        assert len(x_test_flat["input_word_ids"]) == len(y_test_flat)

        # Shuffle data and return
        x_train_flat, y_train_flat = shuffle_bert_data(x_train_flat, y_train_flat)
        return hp, x_train_flat, y_train_flat, x_test_flat, y_test_flat


class BertPlusStatsTweetLevelExperiment(AbstractBertExperiment):
    """
    Train a BERT model on individual tweets, however concatenate tweet-level statistical data to BERT's output
    embedding before feeding it into the dense classifier:
    * Bert input: "<user_i_tweet_j>"
    * dense classifier input: concatenate(<user_i_tweet_j_statistical_data>, <user_i_tweet_j_embedding>)
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
            dropout_rate=hp.Float("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0, 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0, 0.0001, 0.001, 0.01]),
            dense_activity_reg=hp.Choice("Bert.dense_activity_reg", [0, 0.0001, 0.001, 0.01]),
        )

        return self.compile_model_with_adamw(hp, bert_input, dense_out)

    # def cv_data_transformer(self, x_train, y_train, x_test, y_test):

    @classmethod
    def _preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        # Extract tweet-level statistical features
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test)

        # Preprocess data for BERT
        _, x_train_bert_pre, _, x_test_bert_pre, _ = cls.preprocess_data(hp, x_train, y_train, x_test, y_test)

        # Tokenize for BERT
        _, x_train_bert, y_train_bert, x_test_bert, y_test_bert = cls.tokenize_data(
            hp, x_train_bert_pre, y_train, x_test_bert_pre, y_test, tokenizer_class=BertIndividualTweetTokenizer)
        x_train_bert["tweet_level_stats"] = x_train_stats
        x_test_bert["tweet_level_stats"] = x_test_stats

        x_train_bert, y_train_bert = shuffle_bert_data(x_train_bert, y_train_bert)
        return hp, x_train_bert, y_train_bert, x_test_bert, y_test_bert


if __name__ == "__main__":
    """ Execute experiments in this module """
    print("Reading in data")
    dataset_dir = sys.argv[1]
    experiment_dir = sys.argv[2]
    x, y = parse_dataset(dataset_dir, "en")
    print("Beginning experiments")

    with tf.device("/gpu:0"):
        # BertPlusStatsEmbeddingTweetLevelExperiment
        config = {
            "max_trials": 20,
            "hyperparameters": {
                "epochs": 8,
                "batch_size": 64,
                "learning_rate": [2e-5, 5e-5],
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
