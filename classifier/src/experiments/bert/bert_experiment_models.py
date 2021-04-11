import numpy as np
import tensorflow as tf

import data.preprocess as pre
from bert import BertIndividualTweetTokenizer
from bert.models import bert_tokenizer, extract_bert_pooled_output
from experiments.experiment import AbstractBertExperiment, ExperimentConfig
from experiments.models import CompileOnFitKerasModel
from experiments.tuners import GridSearchCV
from statistical.data_extraction import tweet_level_extractor
from statistical.data_extraction.sentiment.vader import VaderSentimentAnalysisWrapper


""" A collection of models and experiments for training/hyperparameter-tuning the BERT-based model """


class BertUserLevelClassifier(tf.keras.layers.Layer):
    """
    A BERT classifier layer, which takes in `Bert.tweet_feed_len` tweets, creates embeddings for each one using
    shared BERT weights and pools the results.
    """

    def __init__(self, hp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters = hp
        self.bert_inputs, _, self.bert_encoder = AbstractBertExperiment.get_bert_layers(
            self.hyperparameters, return_encoder=True)
        if self.hyperparameters.get("selected_encoder_outputs") != "default":
            self.bert_pooled_output_pooling = tf.keras.layers.Dense(
                self.hyperparameters.get("Bert.hidden_size"), activation="tanh")
        else:
            self.bert_pooled_output_pooling = None
        self.pooling = self._make_pooler()
        self.dropout = tf.keras.layers.Dropout(self.hyperparameters.get("Bert.dropout_rate"))
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def _make_pooler(self):
        # Returns a pooler of type: (batch_size, TWEET_FEED_LEN, Bert.hidden_size) => (batch_size, Bert.hidden_size)
        pooler = self.hyperparameters.get("Bert.pooler")

        if pooler == "max":
            return tf.keras.layers.GlobalMaxPool1D()  # shape == (batch_size, Bert.hidden_size)
        elif pooler == "average":
            return tf.keras.layers.GlobalAveragePooling1D()  # shape == (batch_size, Bert.hidden_size)
        elif pooler == "concat":
            return tf.keras.layers.Flatten()  # shape == (batch_size, TWEET_FEED_LEN * Bert.hidden_size)
        else:
            raise ValueError("Invalid value for `Bert.pooler`")

    @tf.function
    def _run_bert_encoder(self, inputs, training=None):
        # inputs.shape == (batch_size, 3, Bert.hidden_size)
        # Returns a Tensor with shape (batch_size, 128)
        bert_input = dict(input_word_ids=inputs[:, 0], input_mask=inputs[:, 1], input_type_ids=inputs[:, 2])
        bert_out = self.bert_encoder(bert_input, training=training)

        selected_encoder_outputs = self.hyperparameters.get("selected_encoder_outputs")
        pooled_output = extract_bert_pooled_output(bert_out, self.bert_pooled_output_pooling, selected_encoder_outputs)
        return tf.reshape(pooled_output, (-1, 1, self.hyperparameters.get("Bert.hidden_size")))

    @tf.function
    def _accumulate_bert_outputs(self, inputs, training=None):
        # inputs.shape == (batch_size, TWEET_FEED_LEN, 3, Bert.hidden_size)
        # Returns a tensor with shape (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        return tf.concat([self._run_bert_encoder(
            inputs[:, i], training=training) for i in range(self.hyperparameters.get("Bert.tweet_feed_len"))], axis=1)

    def call(self, inputs, training=None, mask=None):
        # inputs.shape == (batch_size, TWEET_FEED_LEN, 3)
        # Returns a tensor with shape (batch_size, 1)
        x_train = self._accumulate_bert_outputs(inputs, training=training)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size)

        if self.hyperparameters.get("Bert.tweet_feed_len") > 1:
            x_train = self.pooling(x_train)
            # x_train.shape == (batch_size, Bert.hidden_size) or (batch_size, TWEET_FEED_LEN * Bert.hidden_size)

        if self.hyperparameters.Fixed("Bert.use_batch_norm", False):
            x_train = self.batch_norm(x_train)

        x_train = self.dropout(x_train, training=training)
        return x_train

    def get_config(self):
        pass


class BertPlusStatsUserLevelClassifier(BertUserLevelClassifier):
    """ A BertUserLevelClassifier which also pools tweet-level statistical data """

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


class BertTrainedOnDownstreamLossExperiment(AbstractBertExperiment):
    """
    Train a BERT model where multiple BERT model's with shared weights are used to create embeddings for
    `Bert.tweet_feed_len` tweets, which are then pooled and a classification made.

     * Bert.tweet_feed_len = 1 trains a BERT model on individual tweets
     * Bert.tweet_feed_len = 100 trains a BERT model on the entire tweet feed
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, tuner_class=GridSearchCV)
        self.tuner.num_folds = 3  # 3 CV folds

    @classmethod
    def build_model(cls, hp):
        # BERT tweet chunk pooler
        inputs = tf.keras.layers.Input((hp.get("Bert.tweet_feed_len"), 3, hp.get("Bert.hidden_size")), dtype=tf.int32)
        bert_outputs = BertUserLevelClassifier(hp)(inputs)
        # bert_outputs.shape == (batch_size, Bert.hidden_size)

        # Hidden feed-forward layers
        bert_output_dim = bert_outputs.shape[-1]
        for i in range(hp.Fixed("Bert.num_hidden_layers", 0)):
            dense = tf.keras.layers.Dense(
                bert_output_dim // max(1, i * 2),  # Half the dimension of each feed-forward layer
                activation=hp.get("Bert.hidden_dense_activation"),
                kernel_regularizer=tf.keras.regularizers.l2(hp.get("Bert.dense_kernel_reg")),
            )(bert_outputs)
            if hp.Fixed("Bert.use_batch_norm", False):
                dense = tf.keras.layers.BatchNormalization()(dense)
            dropout = tf.keras.layers.Dropout(hp.get("Bert.dropout_rate"))(dense)
            bert_outputs = dropout

        # Final linear layer
        linear = tf.keras.layers.Dense(
            1,
            activation=hp.Fixed("Bert.dense_activation", "linear"),
            kernel_regularizer=tf.keras.regularizers.l2(hp.get("Bert.dense_kernel_reg")),
        )(bert_outputs)

        return CompileOnFitKerasModel(inputs, linear, optimizer_learning_rate=hp.get("learning_rate"))

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test, shuffle_data=True):
        return cls.tokenize_cv_data(*cls.preprocess_data(hp, x_train, y_train, x_test, y_test), shuffle=shuffle_data)

    @staticmethod
    def tokenize_x(hp, tokenizer, x, shuffle=True):
        def tokenize_tweet_feed(tweet_feed):
            data = [list(tokenizer.tokenize_input([[tweet]]).values()) for tweet in tweet_feed]
            return tf.random.shuffle(data) if shuffle else data

        x_tok = tf.convert_to_tensor(list(map(tokenize_tweet_feed, x)))
        x_chunked = tf.reshape(x_tok, shape=(-1, hp.get("Bert.tweet_feed_len"), 3, hp.get("Bert.hidden_size")))
        # shape(-1, 100, 3, 128) => shape(-1, TWEET_FEED_LEN, 3, 128)
        return x_chunked

    @staticmethod
    def tokenize_y(hp, y):
        return tf.convert_to_tensor([v for v in y for _ in range(100 // hp.get("Bert.tweet_feed_len"))])

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
    stats_preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
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
