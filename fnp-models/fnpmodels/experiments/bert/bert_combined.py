import sys

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fnpmodels.models.bert.tokenize import BertIndividualTweetTokenizer, bert_tokenizer
from fnpmodels.models.sklearn import VotingClassifier
from fnpmodels.experiments.experiment import AbstractSklearnExperiment
from fnpmodels.experiments.handler import ExperimentHandler
from .bert_experiment_models import BertTrainedOnDownstreamLossExperiment


class BertDownstreamLossLogitsCombinedExperiment(AbstractSklearnExperiment):

    @classmethod
    def build_model(cls, hp):
        if hp.get("Sklearn.model_type") == VotingClassifier.__name__:
            return VotingClassifier()

        return cls.select_sklearn_model(hp)

    @classmethod
    def cv_data_transformer(cls, hp, x_train, y_train, x_test, y_test):
        bert_model = cls.fit_bert(hp, x_train, y_train, x_test, y_test)
        x_train, y_train, x_test, y_test = cls.predict_user_data(hp, bert_model, x_train, y_train, x_test, y_test)
        x_train = tf.math.sigmoid(x_train)
        x_test = tf.math.sigmoid(x_test)
        x_train, x_test = cls.pool_combined(hp, x_train, x_test)
        return cls.to_float64(x_train, y_train, x_test, y_test)

    @staticmethod
    def to_float64(x_train, y_train, x_test, y_test):
        # RandomForestClassifier doesn't support float32
        def cast(data):
            return np.asarray(tf.cast(data, tf.float64))

        return cast(x_train), cast(y_train), cast(x_test), cast(y_test)

    @staticmethod
    def pool_combined(hp, x_train, x_test):
        # x_.shape == (num_users, TWEET_FEED_LEN, -1)
        pooler = hp.get("Combined.pooler")
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

    @staticmethod
    def fit_bert(hp, x_train, y_train, x_test, y_test):
        # Preprocess BERT data
        _, x_train_bert, y_train_bert, x_test_bert, y_test_bert = BertTrainedOnDownstreamLossExperiment.preprocess_cv_data(
            hp, x_train, y_train, x_test, y_test)

        # Train BERT model
        bert_model = BertTrainedOnDownstreamLossExperiment.build_model(hp)
        bert_model.fit(
            x=x_train_bert,
            y=y_train_bert,
            epochs=hp.get("Bert.epochs"),
            batch_size=hp.get("Bert.batch_size"),
            validation_data=(x_test_bert, y_test_bert),
        )
        return bert_model

    @staticmethod
    def predict_user_data(hp, bert_model, x_train, y_train, x_test, y_test):
        # Preprocess user-level data and predict using BERT
        tokenizer = bert_tokenizer(
            hp.get("Bert.encoder_url"),
            hp.get("Bert.hidden_size"),
            BertIndividualTweetTokenizer
        )

        def predict_data(x):
            return np.asarray([
                bert_model.predict(
                    BertTrainedOnDownstreamLossExperiment.tokenize_x(hp, tokenizer, [tweet_feed])
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

    @classmethod
    def build_model(cls, hp):
        """ Build an Sklearn pipeline with a final estimator """
        if hp.Fixed("Sklearn.scale", False):
            estimator = cls.select_sklearn_model(hp)
            steps = [("scaler", StandardScaler()), ("estimator", estimator)]

            # Add PCA to deal with multi-colinearity issues in data (Gradient Boosting doesn't have this issue)
            if hp.get("Sklearn.model_type") != "XGBClassifier":
                steps.insert(0, ("PCA", PCA()))

            return Pipeline(steps)

        return cls.select_sklearn_model(hp)

    @classmethod
    def cv_data_transformer(cls, hp, x_train, y_train, x_test, y_test):
        bert_model = cls.fit_bert(hp, x_train, y_train, x_test, y_test)
        bert_pooled_model = tf.keras.Model(bert_model.inputs, bert_model.layers[-2].output)
        x_train, y_train, x_test, y_test = cls.predict_user_data(
            hp, bert_pooled_model, x_train, y_train, x_test, y_test)
        x_train, x_test = cls.pool_combined(hp, x_train, x_test)
        return cls.to_float64(x_train, y_train, x_test, y_test)

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return hp, x_train, y_train, x_test, y_test


def logits_experiment_handler(bert_hp_dict: dict):
    """ Trains a BERT model and then fits a secondary model to predict based on BERT's output logits"""
    experiments = [
        (
            # Classifying BERT output logits
            BertDownstreamLossLogitsCombinedExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss/combined",
                "experiment_name": "logits",
                "max_trials": 100,
                "hyperparameters": {
                    "Combined.pooler": "concat",
                    "Sklearn.model_type": ["VotingClassifier", "LogisticRegression", "SVC",
                                           "RandomForestClassifier", "XGBClassifier"],
                    **bert_hp_dict,
                },
            }
        )
    ]
    return ExperimentHandler(experiments)


def pooling_experiment_handler(bert_hp_dict: dict):
    """ Trains a BERT model and then fits a secondary model to predict based on BERT's final hidden layer """
    pooler_args = ["concat", "max", "average"]
    experiments = [
        (
            BertDownstreamLossPooledCombinedExperiment,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss/combined/pooler",
                "experiment_name": pooler,
                "max_trials": 100,
                "hyperparameters": {
                    "Combined.pooler": pooler,
                    **bert_hp_dict,
                },
            }
        ) for pooler in pooler_args
    ]
    return ExperimentHandler(experiments)


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    # BERT chunked model
    bert_hp = {
        "learning_rate": 2e-5,
        "Bert.epochs": 4,
        "Bert.batch_size": 8,
        "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        "Bert.hidden_size": 128,
        "selected_encoder_outputs": "default",
        "Bert.dense_activation": "linear",
        "Bert.pooler": "max",
        "Bert.preprocessing": "[replace_emojis_no_sep, remove_tags]",
        "Bert.dropout_rate": 0.1,
        "Bert.dense_kernel_reg": 0.0001,
        "Bert.use_batch_norm": False,
        "Bert.num_hidden_layers": 0,
        "Bert.tweet_feed_len": 10,
    }

    with tf.device("/gpu:0"):
        print("BERT chunked experiments with tweet_feed_len of", bert_hp["Bert.tweet_feed_len"])
        # Logits experiment
        handler = logits_experiment_handler(bert_hp)
        # handler.run_experiments(dataset_dir)
        handler.print_results(10)

        # Pooler experiments
        handler = pooling_experiment_handler(bert_hp)
        # handler.run_experiments(dataset_dir)
        handler.print_results(10)

    # BERT individual model
    bert_hp = {
        "learning_rate": 2e-5,
        "Bert.epochs": 4,
        "Bert.batch_size": 8,
        "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        "Bert.hidden_size": 128,
        "selected_encoder_outputs": "default",
        "Bert.dense_activation": "linear",
        "Bert.pooler": "max",
        "Bert.preprocessing": "[replace_emojis_no_sep, remove_tags]",
        "Bert.dropout_rate": 0.1,
        "Bert.dense_kernel_reg": 0.0001,
        "Bert.use_batch_norm": False,
        "Bert.num_hidden_layers": 0,
        "Bert.tweet_feed_len": 1,
    }

    with tf.device("/gpu:0"):
        print("BERT individual experiments with tweet_feed_len of", bert_hp["Bert.tweet_feed_len"])
        # Logits experiment
        handler = logits_experiment_handler(bert_hp)
        # handler.run_experiments(dataset_dir)
        handler.print_results(10)

        # Pooler experiments
        handler = pooling_experiment_handler(bert_hp)
        # handler.run_experiments(dataset_dir)
        handler.print_results(10)
