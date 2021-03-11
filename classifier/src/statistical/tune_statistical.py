import math
import numpy as np

import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

from kerastuner import HyperParameters, Objective
from kerastuner.oracles import BayesianOptimization
from kerastuner.tuners import Sklearn

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from statistical.data_extraction import readability_tweet_extractor, ner_tweet_extractor, sentiment_tweet_extractor


class SklearnTunerPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        step_names = set(name for name, _ in self.steps)
        updated_fit_params = {}

        for name, param in fit_params.items():
            if name.split("__")[0] in step_names:
                updated_fit_params[name] = param
            else:
                updated_fit_params[self.steps[-1][0] + "__" + name] = param

        super().fit(X, y, **updated_fit_params)


def build_sklearn_classifier_model(hps: HyperParameters) -> BaseEstimator:
    """ Build an SkLearn classifier """
    sklearn_model = hps.Choice(
        "sklearn_model",
        [LogisticRegression.__name__, SVC.__name__, RandomForestClassifier.__name__, XGBClassifier.__name__]
    )

    if sklearn_model == LogisticRegression.__name__:
        with hps.conditional_scope("sklearn_model", LogisticRegression.__name__):
            estimator = LogisticRegression(
                C=hps.Float("LogisticRegression_C", 0.0001, 1000),
            )
    elif sklearn_model == SVC.__name__:
        with hps.conditional_scope("sklearn_model", SVC.__name__):
            estimator = SVC(
                C=hps.Float("SVC_C", 0.0001, 1000),
                probability=True,
            )
    elif sklearn_model == RandomForestClassifier.__name__:
        with hps.conditional_scope("sklearn_model", RandomForestClassifier.__name__):
            estimator = RandomForestClassifier(
                n_estimators=hps.Choice("RandomForestClassifier_n_estimators", [100, 200, 300, 400, 500, 600]),
                min_samples_split=hps.Int("RandomForestClassifier_min_samples_split", 2, 10),
                min_samples_leaf=hps.Int("RandomForestClassifier_min_samples_leaf", 1, 10),
            )
    elif sklearn_model == XGBClassifier.__name__:
        with hps.conditional_scope("sklearn_model", XGBClassifier.__name__):
            estimator = XGBClassifier(
                learning_rate=hps.Float("XGBClassifier_learning_rate", 0.0001, 0.3),
                gamma=hps.Int("XGBClassifier_gamma", 0, 10),
                max_depth=hps.Int("XGBClassifier_max_depth", 1, 8),
                min_child_weight=hps.Int("XGBClassifier_min_child_weight", 0, 10),
                subsample=hps.Float("XGBClassifier_subsample", 0, 1),
                colsample_bytree=hps.Float("XGBClassifier_colsample_bytree", 0.2, 1),
                colsample_bylevel=hps.Float("XGBClassifier_colsample_bylevel", 0.2, 1),
                colsample_bynode=hps.Float("XGBClassifier_colsample_bynode", 0.2, 1),
                reg_lambda=hps.Float("XGBClassifier_reg_lambda", 0.001, 0.5),
                reg_alpha=hps.Float("XGBClassifier_reg_alpha", 0.001, 0.5),
            )
    else:
        raise RuntimeError("Invalid SkLearn model name")

    return SklearnTunerPipeline([("PCA", PCA()), ("scaler", StandardScaler()), ("estimator", estimator)])


def build_nn_classifier_model(hps: HyperParameters) -> Model:
    """ Build a neural network classifier """
    data_len = hps.get("input_data_len")

    inputs = Input((data_len,))
    if hps.Boolean("use_relu"):
        with hps.conditional_scope("use_relu", True):
            inputs = Dense(
                data_len,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(hps.Float("relu_kernel_reg", 0.0001, 0.1)),
                bias_regularizer=tf.keras.regularizers.l2(hps.Float("relu_bias_reg", 0.0001, 0.1)),
            )(inputs)
            batch = BatchNormalization()(inputs)
            inputs = Dropout(hps.Float("dropout_rate", 0, 0.5))(batch)
    linear = Dense(
        1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(hps.Float("linear_kernel_reg", 0.0001, 0.1)),
        bias_regularizer=tf.keras.regularizers.l2(hps.Float("linear_bias_reg", 0.0001, 0.1)),
    )(inputs)

    model = Model(inputs, linear)
    # TODO model.compile()
    # TODO setup separate keras tuner for this for tune_combined_statistical_model
    return model


def tune_readability_model(x_train, y_train, project_name, **kwargs):
    return tune_sklearn_model(x_train, y_train, "readability_" + project_name, readability_tweet_extractor(), **kwargs)


def tune_ner_model(x_train, y_train, project_name, **kwargs):
    return tune_sklearn_model(x_train, y_train, "ner_" + project_name, ner_tweet_extractor(), **kwargs)


def tune_sentiment_model(x_train, y_train, project_name, **kwargs):
    return tune_sklearn_model(x_train, y_train, "sentiment_" + project_name, sentiment_tweet_extractor(), **kwargs)


def tune_combined_statistical_models(x_train, y_train, project_name, **kwargs):
    x_train_readability = readability_tweet_extractor().transform(x_train)
    x_train_ner = ner_tweet_extractor().transform(x_train)
    x_train_sentiment = sentiment_tweet_extractor().transform(x_train)
    x_train = np.concatenate([x_train_readability, x_train_ner, x_train_sentiment], axis=1)
    print(x_train.shape)

    return tune_sklearn_model(x_train, y_train, "combined_" + project_name, **kwargs)


def tune_sklearn_model(x_train, y_train, project_name, feature_extractor=None, **kwargs):
    # Extract features
    if feature_extractor is not None:
        x_train = feature_extractor.transform(x_train)

    # Setup Keras Tuner
    tuner = sklearn_tuner(project_name, **kwargs)
    tuner.search(x_train, y_train)
    return tuner


def sklearn_tuner(project_name, max_trials=30, directory="../../training/statistical"):
    return Sklearn(
        oracle=BayesianOptimization(
            objective=Objective("score", "min"),  # minimise log loss
            max_trials=max_trials,
        ),
        hypermodel=build_sklearn_classifier_model,
        scoring=make_scorer(loss_accuracy_scorer, needs_proba=True),
        metrics=[accuracy_score, f1_score],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3),
        directory=directory,
        project_name=project_name,
    )


def loss_accuracy_scorer(y_true, y_pred, **kwargs):
    loss = log_loss(y_true, y_pred, **kwargs)
    accuracy = accuracy_score(y_true, np.round(y_pred), **kwargs)
    return math.log(loss + 1) / math.log(accuracy + 1)
