import numpy as np
import tensorflow as tf

import processing.preprocess as pre
from processing.statistical.tweet_level import tweet_level_extractor
from processing.statistical.sentiment.vader import VaderSentimentAnalysisWrapper
from models.bert.tokenize import BertIndividualTweetTokenizer, bert_tokenizer, tokenize_x, tokenize_y
from models.bert.models import BertPlusStatsUserLevelClassifier, build_bert_model
from experiments.experiment import AbstractBertExperiment, ExperimentConfig
from experiments.models import CompileOnFitKerasModel
from experiments.tuners import GridSearchCV


""" A collection of models and experiments for training/hyperparameter-tuning the BERT-based model """


class BertTrainedOnDownstreamLossExperiment(AbstractBertExperiment):
    """
    Train a BERT model where multiple BERT model's with shared weights are used to create embeddings for
    `Bert.tweet_feed_len` tweets, which are then pooled and a classification made.

     * Bert.tweet_feed_len = 1 trains a BERT model on individual tweets
     * Bert.tweet_feed_len = 100 trains a BERT model on the entire tweet feed
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, tuner_class=GridSearchCV)

    @classmethod
    def build_model(cls, hp):
        return build_bert_model(hp)

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test, shuffle_data=True):
        return cls.tokenize_cv_data(*cls.preprocess_data(hp, x_train, y_train, x_test, y_test), shuffle=shuffle_data)

    @staticmethod
    def tokenize_x(hp, tokenizer, x, shuffle=True):
        return tokenize_x(hp, tokenizer, x, shuffle=shuffle)

    @staticmethod
    def tokenize_y(hp, y):
        return tokenize_y(hp, y)

    @classmethod
    def tokenize_cv_data(cls, hp, x_train, y_train, x_test, y_test, shuffle=True):
        tokenizer = bert_tokenizer(hp.get("Bert.encoder_url"), hp.get("Bert.hidden_size"), BertIndividualTweetTokenizer)

        def tokenize(x, y, shuffle_seed=1):
            # Tokenize data
            x_tok = cls.tokenize_x(hp, tokenizer, x, shuffle=shuffle)
            y_tok = cls.tokenize_y(hp, y)

            # Shuffle data
            if shuffle:
                tf.random.set_seed(shuffle_seed)
                x_tok = tf.random.shuffle(x_tok, seed=shuffle_seed)
                tf.random.set_seed(shuffle_seed)
                y_tok = tf.random.shuffle(y_tok, seed=shuffle_seed)

            return x_tok, y_tok

        x_train, y_train = tokenize(x_train, y_train)
        x_test = cls.tokenize_x(hp, tokenizer, x_test)
        y_test = cls.tokenize_y(hp, y_test)
        # x_.shape == (num_users * 100/TWEET_FEED_LEN, TWEET_FEED_LEN, 3, Bert.hidden_size)
        # y_.shape == (num_users * 100/TWEET_FEED_LEN)
        return hp, x_train, y_train, x_test, y_test


def extract_tweet_level_stats(x_train, x_test, extractor):
    """ Extract tweet-level statistical data """
    stats_preprocessor = pre.TweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
    x_train_stats_pre = stats_preprocessor.transform(x_train)
    x_test_stats_pre = stats_preprocessor.transform(x_test)

    x_train_stats = tf.convert_to_tensor(list(map(extractor.transform, x_train_stats_pre)))
    x_test_stats = tf.convert_to_tensor(list(map(extractor.transform, x_test_stats_pre)))
    return x_train_stats, x_test_stats


class BertPlusStatsExperiment(BertTrainedOnDownstreamLossExperiment):
    """ A BertTrainedOnDownstreamLossExperiment where the classifier also uses tweet-level statistical data """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.stats_extractor = tweet_level_extractor(VaderSentimentAnalysisWrapper())

    def build_model(self, hp):
        bert_input = tf.keras.layers.Input((hp.get("Bert.tweet_feed_len"), 3, hp.get("Bert.hidden_size")),
                                           dtype=tf.int32, name="BERT_input")
        stats_input = tf.keras.layers.Input((hp.get("Bert.tweet_feed_len"), len(self.stats_extractor.feature_names)),
                                            dtype=tf.float32, name="stats_input")
        inputs = [bert_input, stats_input]

        # BERT model
        bert_outputs = BertPlusStatsUserLevelClassifier(hp)(inputs)

        # Final linear layer
        linear = tf.keras.layers.Dense(
            1,
            activation=hp.Fixed("Bert.dense_activation", "linear"),
            kernel_regularizer=tf.keras.regularizers.l2(hp.get("Bert.dense_kernel_reg")),
        )(bert_outputs)

        return CompileOnFitKerasModel(inputs, linear, optimizer_learning_rate=hp.get("learning_rate"))

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        # Tokenize and preprocess BERT data
        _, x_train_bert, y_train, x_test_bert, y_test = super().preprocess_cv_data(
            self.hyperparameters, x_train, y_train, x_test, y_test, shuffle_data=False)

        # Extract tweet-level statistical data
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test, self.stats_extractor)

        def process_stats(x):
            return tf.reshape(
                tf.cast(x, dtype=tf.float32),
                shape=(-1, self.hyperparameters.get("Bert.tweet_feed_len"), len(self.stats_extractor.feature_names))
            )

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
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test, **kwargs):
        return hp, x_train, y_train, x_test, y_test


class BertPlusStatsEmbeddingExperiment(BertTrainedOnDownstreamLossExperiment):
    """
    A BertTrainedOnDownstreamLossExperiment where tweets are concatenated with tweet-level statistics of the form:
    "<stat_1> <stat_2> ... <stat_n> | <tweet>"
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.stats_extractor = tweet_level_extractor(VaderSentimentAnalysisWrapper())

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        # Extract tweet-level statistical features
        x_train_stats, x_test_stats = extract_tweet_level_stats(x_train, x_test, self.stats_extractor)

        # Pre-process tweet data for BERT
        _, x_train_pre, _, x_test_pre, _ = self.preprocess_data(self.hyperparameters, x_train, y_train, x_test, y_test)

        # Combine BERT tweet strings with statistical data ("<tweet>" --> "<stat_1> ... <stat_n> | <tweet>")
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
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test, **kwargs):
        return hp, x_train, y_train, x_test, y_test
