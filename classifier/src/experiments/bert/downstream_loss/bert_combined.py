import sys

import numpy as np
import tensorflow as tf

from bert import BertIndividualTweetTokenizer
from bert.models import bert_tokenizer
from experiments.bert.downstream_loss.bert_downstream_loss import BertTrainedOnDownstreamLoss
from experiments.experiment import AbstractSklearnExperiment
from experiments.handler import ExperimentHandler


class BertDownstreamLossLogitsCombinedExperiment(AbstractSklearnExperiment):

    def build_model(self, hp):
        return AbstractSklearnExperiment.select_sklearn_model(hp)

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        bert_model = self.fit_bert(x_train, y_train, x_test, y_test)
        x_train, y_train, x_test, y_test = self.predict_user_data(bert_model, x_train, y_train, x_test, y_test)
        x_train, x_test = self.pool_combined(x_train, x_test)
        return x_train, y_train, x_test, y_test

    def pool_combined(self, x_train, x_test):
        # x_.shape == (num_users, TWEET_FEED_LEN, -1)
        pooler = self.hyperparameters.get("Combined.pooler")
        if pooler == "concat":
            x_train = tf.keras.layers.Flatten()(x_train)
            x_test = tf.keras.layers.Flatten()(x_test)
        elif pooler == "max":
            x_train = tf.keras.layers.GlobalMaxPool1D()(x_train)
            x_test = tf.keras.layers.GlobalMaxPool1D()(x_test)
        elif pooler == "average":
            x_train = tf.keras.layers.GlobalAveragePooling1D()(x_train)
            x_test = tf.keras.layers.GlobalAveragePooling1D()(x_test)
        else:
            raise ValueError("Invalid value for `Combined.pooler`")

        return x_train, x_test

    def fit_bert(self, x_train, y_train, x_test, y_test):
        # Preprocess BERT data
        _, x_train_bert, y_train_bert, x_test_bert, y_test_bert = BertTrainedOnDownstreamLoss.preprocess_cv_data(
            self.hyperparameters, x_train, y_train, x_test, y_test)

        # Train BERT model
        bert_model = BertTrainedOnDownstreamLoss.build_model(None, self.hyperparameters)
        bert_model.fit(
            x=x_train_bert,
            y=y_train_bert,
            epochs=self.hyperparameters.get("Bert.epochs"),
            batch_size=self.hyperparameters.get("Bert.batch_size"),
            validation_data=(x_test_bert, y_test_bert),
        )
        return bert_model

    def predict_user_data(self, bert_model, x_train, y_train, x_test, y_test):
        # Preprocess user-level data and predict using BERT
        tokenizer = bert_tokenizer(
            self.hyperparameters.get("Bert.encoder_url"),
            self.hyperparameters.get("Bert.hidden_size"),
            BertIndividualTweetTokenizer
        )

        def predict_data(x):
            return np.asarray([
                bert_model.predict(
                    BertTrainedOnDownstreamLoss.tokenize_x(self.hyperparameters, tokenizer, [tweet_feed])
                ) for tweet_feed in x
            ])  # (num_users, TWEET_FEED_LEN, -1)

        x_train = predict_data(x_train)
        x_test = predict_data(x_test)
        # x_.shape == (num_users, TWEET_FEED_LEN, -1)

        return x_train, y_train, x_test, y_test

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return hp, x_train, y_train, x_test, y_test


class BertDownstreamLossPooledCombinedExperiment(BertDownstreamLossLogitsCombinedExperiment):

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        bert_model = self.fit_bert(x_train, y_train, x_test, y_test)
        bert_pooled_model = tf.keras.Model(bert_model.inputs, bert_model.layers[-2].output)
        x_train, y_train, x_test, y_test = self.predict_user_data(bert_pooled_model, x_train, y_train, x_test, y_test)
        x_train, x_test = self.pool_combined(x_train, x_test)
        return x_train, y_train, x_test, y_test

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return hp, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            BertDownstreamLossLogitsCombinedExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss_logits_combined",
                "experiment_name": "indiv_1",
                "max_trials": 100,
                "hyperparameters": {
                    "learning_rate": 5e-5,
                    "Bert.epochs": 2,
                    "Bert.batch_size": 8,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags]",
                    "selected_encoder_outputs": "sum_last_4_hidden_layers",
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_activation": "linear",
                    "Bert.pooler": "concat",
                    "Bert.dense_kernel_reg": 0.,
                    "Bert.dense_bias_reg": 0.,
                    "Combined.pooler": "concat",
                },
            }
        ), (
            BertDownstreamLossPooledCombinedExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss_pooled_combined",
                "experiment_name": "concat_pooler",
                "max_trials": 100,
                "hyperparameters": {
                    "learning_rate": 5e-5,
                    "Bert.epochs": 6,
                    "Bert.batch_size": 8,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags]",
                    "selected_encoder_outputs": "sum_last_4_hidden_layers",
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_activation": "linear",
                    "Bert.pooler": "concat",
                    "Bert.dense_kernel_reg": 0.,
                    "Bert.dense_bias_reg": 0.,
                    "Combined.pooler": "concat",
                },
            }
        ), (
            BertDownstreamLossPooledCombinedExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss_pooled_combined",
                "experiment_name": "max_pooler",
                "max_trials": 100,
                "hyperparameters": {
                    "learning_rate": 5e-5,
                    "Bert.epochs": 6,
                    "Bert.batch_size": 8,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags]",
                    "selected_encoder_outputs": "sum_last_4_hidden_layers",
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_activation": "linear",
                    "Bert.pooler": "concat",
                    "Bert.dense_kernel_reg": 0.,
                    "Bert.dense_bias_reg": 0.,
                    "Combined.pooler": "max",
                },
            }
        ), (
            BertDownstreamLossPooledCombinedExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss_pooled_combined",
                "experiment_name": "average_pooler",
                "max_trials": 100,
                "hyperparameters": {
                    "learning_rate": 5e-5,
                    "Bert.epochs": 6,
                    "Bert.batch_size": 8,
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": "[remove_emojis, remove_tags]",
                    "selected_encoder_outputs": "sum_last_4_hidden_layers",
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_activation": "linear",
                    "Bert.pooler": "concat",
                    "Bert.dense_kernel_reg": 0.,
                    "Bert.dense_bias_reg": 0.,
                    "Combined.pooler": "average",
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        # handler.run_experiments(dataset_dir)
        handler.print_results(20)
