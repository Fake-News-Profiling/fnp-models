import sys

import tensorflow as tf

from bert import BertIndividualTweetTokenizer
from bert.models import bert_tokenizer
from experiments.experiment import AbstractBertExperiment
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel


class BertUserLevelClassifier(tf.keras.layers.Layer):
    def __init__(self, hp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert_inputs, _, self.bert_encoder = AbstractBertExperiment.get_bert_layers(hp, return_encoder=True)
        self.pooling = tf.keras.layers.GlobalMaxPool1D()
        self.dropout = tf.keras.layers.Dropout(hp.Fixed("Bert.dropout_rate", 0.1))
        self.linear = tf.keras.layers.Dense(1, activation=hp.Fixed("Bert.dense_activation", "linear"))
        self.hyperparameters = hp

    @tf.function
    def _run_bert_encoder(self, inputs, training=None):
        # inputs.shape == (batch_size, 3, Bert.hidden_size)
        # Returns a Tensor with shape (batch_size, 128)
        return tf.reshape(self.bert_encoder(
            dict(input_word_ids=inputs[:, 0], input_mask=inputs[:, 1], input_type_ids=inputs[:, 2]),
            training=training,
        )["pooled_output"], (-1, 1, self.hyperparameters.get("Bert.hidden_size")))

    @tf.function
    def _accumulate_bert_outputs(self, inputs, training=None):
        # inputs.shape == (batch_size, 100, 3)
        # Returns a tensor with shape (batch_size, num_tweets_per_user, Bert.hidden_size)
        num_tweets_per_user = 100
        return tf.concat([self._run_bert_encoder(
            inputs[:, i], training=training) for i in range(num_tweets_per_user)], axis=1)

    def call(self, inputs, training=None, mask=None):
        # inputs.shape == (batch_size, 100, 3)
        # Returns a tensor with shape (batch_size, 1)
        x_train = self._accumulate_bert_outputs(inputs, training=training)
        # x_train.shape == (batch_size, num_tweets_per_user, Bert.hidden_size)
        x_train = self.pooling(x_train)
        # x_train.shape == (batch_size, Bert.hidden_size)
        x_train = self.dropout(x_train, training=training)
        return self.linear(x_train)

    def get_config(self):
        pass


class BertTrainedOnDownstreamLoss(AbstractBertExperiment):
    """ BERT model trained on individual tweets, however where the loss used is from the overall user classification """

    def build_model(self, hp):
        num_tweets_per_user = 100
        bert_clf = BertUserLevelClassifier(hp)
        inputs = tf.keras.layers.Input((100, 3, 128), dtype=tf.int32)
        # [list(bert_clf.bert_inputs.copy().values()) for _ in range(num_tweets_per_user)]
        outputs = bert_clf(inputs)
        return CompileOnFitKerasModel(inputs, outputs, optimizer_learning_rate=hp.get("learning_rate"))

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        tokenizer = bert_tokenizer(
            hp.get("Bert.encoder_url"), hp.get("Bert.hidden_size"), BertIndividualTweetTokenizer)
        hp, x_train, y_train, x_test, y_test = cls.preprocess_data(hp, x_train, y_train, x_test, y_test)
        num_tweets_per_user = 100

        def tokenize(x):
            return tf.reshape(
                tf.convert_to_tensor(
                    [[list(tokenizer.tokenize_input([[tweet]]).values()) for tweet in tweet_feed] for tweet_feed in x]
                ), (-1, num_tweets_per_user, 3, hp.get("Bert.hidden_size")))

        x_train = tokenize(x_train)
        x_test = tokenize(x_test)
        # x_.shape == (-1, num_tweets_per_user, 3, Bert.hidden_size)
        return hp, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    # Best preprocessing functions found from BertTweetLevelExperiment
    preprocessing_choices = [
        "[remove_emojis, remove_tags]",
        "[remove_emojis, remove_tags, remove_punctuation]",
        "[replace_emojis_no_sep, remove_tags]",
        "[replace_emojis_no_sep, remove_tags, remove_punctuation]",
    ]

    experiments = [
        (
            # Bert (128) Individual with different pooled output methods
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "indiv_1",
                "max_trials": 10,
                "hyperparameters": {
                    "epochs": 4,
                    "batch_size": 32,
                    "learning_rate": 2e-5,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags, remove_punctuation]",
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        handler.run_experiments(dataset_dir)
