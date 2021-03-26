import sys

import tensorflow as tf

from bert import BertIndividualTweetTokenizer
from bert.models import bert_tokenizer, extract_bert_pooled_output
from experiments.experiment import AbstractBertExperiment, ExperimentConfig
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel
from experiments.tuners import GridSearchCV

TWEET_FEED_LEN = 25


class BertUserLevelClassifier(tf.keras.layers.Layer):
    def __init__(self, hp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters = hp
        self.bert_inputs, _, self.bert_encoder = AbstractBertExperiment.get_bert_layers(
            self.hyperparameters, return_encoder=True)
        self.bert_pooled_output_pooling = tf.keras.layers.Dense(
            self.hyperparameters.get("Bert.hidden_size"), activation="tanh")
        self.pooling = self._make_pooler()
        self.dropout = tf.keras.layers.Dropout(self.hyperparameters.Fixed("Bert.dropout_rate", 0.1))
        self.linear = tf.keras.layers.Dense(1, activation=self.hyperparameters.Fixed("Bert.dense_activation", "linear"))

    def _make_pooler(self):
        # Returns a pooler which handles input of shape (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        pooler = self.hyperparameters.Choice("Bert.pooler", ["max", "average", "concat"])

        if pooler == "max":
            return tf.keras.layers.GlobalMaxPool1D()
        elif pooler == "average":
            return tf.keras.layers.GlobalAveragePooling1D()
        elif pooler == "concat":
            return tf.keras.layers.Concatenate()
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
        # inputs.shape == (batch_size, TWEET_FEED_LEN, 3)
        # Returns a tensor with shape (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        return tf.concat([self._run_bert_encoder(
            inputs[:, i], training=training) for i in range(TWEET_FEED_LEN)], axis=1)

    def call(self, inputs, training=None, mask=None):
        # inputs.shape == (batch_size, TWEET_FEED_LEN, 3)
        # Returns a tensor with shape (batch_size, 1)
        x_train = self._accumulate_bert_outputs(inputs, training=training)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        x_train = self.pooling(x_train)
        # x_train.shape == (batch_size, Bert.hidden_size)
        x_train = self.dropout(x_train, training=training)
        return self.linear(x_train)

    def get_config(self):
        pass


class BertTrainedOnDownstreamLoss(AbstractBertExperiment):
    """ BERT model trained on individual tweets, however where the loss used is from the overall user classification """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, tuner_class=GridSearchCV)
        self.tuner.num_folds = 3  # Only train/test using 3 of the 5 folds

    def build_model(self, hp):
        bert_clf = BertUserLevelClassifier(hp)
        inputs = tf.keras.layers.Input((TWEET_FEED_LEN, 3, hp.get("Bert.hidden_size")), dtype=tf.int32)
        outputs = bert_clf(inputs)
        return CompileOnFitKerasModel(inputs, outputs, optimizer_learning_rate=hp.get("learning_rate"))

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return cls.tokenize_cv_data(*cls.preprocess_data(hp, x_train, y_train, x_test, y_test))

    @classmethod
    def tokenize_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        tokenizer = bert_tokenizer(hp.get("Bert.encoder_url"), hp.get("Bert.hidden_size"), BertIndividualTweetTokenizer)

        def tokenize_x(x):
            x_tok = tf.convert_to_tensor(
                [
                    # Shuffle each users tweet feed
                    tf.random.shuffle([list(tokenizer.tokenize_input([[tweet]]).values()) for tweet in tweet_feed])
                    for tweet_feed in x])
            x_chunked = tf.reshape(x_tok, shape=(-1, TWEET_FEED_LEN, 3, hp.get("Bert.hidden_size")))
            # shape(-1, 100, 3, 128) => shape(-1, TWEET_FEED_LEN, 3, 128)
            return x_chunked

        def tokenize_y(y):
            return tf.convert_to_tensor([v for v in y for _ in range(100 // TWEET_FEED_LEN)])

        def tokenize(x, y, shuffle_seed=1):
            # Tokenize data
            x_tok = tokenize_x(x)
            y_tok = tokenize_y(y)

            # Shuffle data
            tf.random.set_seed(shuffle_seed)
            x_shuffled = tf.random.shuffle(x_tok, seed=shuffle_seed)
            tf.random.set_seed(shuffle_seed)
            y_shuffled = tf.random.shuffle(y_tok, seed=shuffle_seed)
            return x_shuffled, y_shuffled

        x_train, y_train = tokenize(x_train, y_train)
        x_test = tokenize_x(x_test)
        y_test = tokenize_y(y_test)
        # x_.shape == (num_users * 100/TWEET_FEED_LEN, TWEET_FEED_LEN, 3, Bert.hidden_size)
        # y_.shape == (num_users * 100/TWEET_FEED_LEN)
        return hp, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    # Best preprocessing functions found from BertTweetLevelExperiment
    preprocessing_choices = [
        "[remove_emojis, remove_tags]",
        "[remove_emojis, remove_tags, remove_punctuation]",
        "[replace_emojis_no_sep, remove_tags]",
    ]
    encoder_output_choices = [
        "default",
        "2nd_to_last_hidden_layer",
        "sum_all_hidden_layers",
        "sum_last_4_hidden_layers",
        "concat_last_4_hidden_layers",
    ]

    experiments = [
        (
            # Bert (128) Individual with different pooled output methods
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "indiv_1",
                "max_trials": 36,
                "hyperparameters": {
                    "epochs": 16,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": preprocessing_choices,
                    "selected_encoder_outputs": encoder_output_choices,
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        handler.run_experiments(dataset_dir)
