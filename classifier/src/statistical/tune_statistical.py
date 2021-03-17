from functools import partial

import numpy as np

import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

from kerastuner import HyperParameters, Objective
from kerastuner.oracles import BayesianOptimization
from kerastuner.tuners import Sklearn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.python.keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard
from xgboost import XGBClassifier

from base import load_hyperparameters
from bert_classifier import BayesianOptimizationCVTunerWithFitHyperParameters
from statistical.data_extraction import readability_tweet_extractor, ner_tweet_extractor, sentiment_tweet_extractor, \
    combined_tweet_extractor, tweet_level_extractor


class SklearnTunerPipeline(Pipeline):
    def __init__(self, steps, tweet_level):
        super().__init__(steps)
        self.tweet_level = tweet_level

    def _to_tweet_level(self, X, y):
        if self.tweet_level:
            if y is not None:
                y = np.asarray([v for v in y for _ in range(len(X[0]))])
            X = np.asarray([tweet for tweet_feed in X for tweet in tweet_feed])

        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        X_tl, y_tl = self._to_tweet_level(X, y)
        return super().fit_transform(X_tl, y_tl, **fit_params)

    def fit(self, X, y=None, **fit_params):
        step_names = set(name for name, _ in self.steps)
        updated_fit_params = {}

        for name, param in fit_params.items():
            if name.split("__")[0] in step_names:
                updated_fit_params[name] = param
            else:
                updated_fit_params[self.steps[-1][0] + "__" + name] = param

        X_tl, y_tl = self._to_tweet_level(X, y)
        return super().fit(X_tl, y_tl, **updated_fit_params)

    def predict(self, X, **predict_params):
        X_tl, _ = self._to_tweet_level(X, None)
        return super().predict(X_tl, **predict_params)

    def _transform(self, X):
        X_tl, _ = self._to_tweet_level(X, None)
        return super()._transform(X_tl)


class PipelineEstimatorWrapper(TransformerMixin, BaseEstimator):
    """ Wrapper for an Sklearn estimator which can be used as an intermediate step in an Sklearn Pipeline """
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        Xt = np.asarray(self.estimator.predict(X)).reshape((-1, 100))
        return Xt

    def predict(self, X):
        return self.estimator.predict(X)


def _stats_wrapper(x_train, y_train, x_test, model_extractors_and_paths):
    x_train_out = []
    x_test_out = []
    for (extractor, path) in model_extractors_and_paths:
        x_train_model = extractor.transform(x_train)
        x_test_model = extractor.transform(x_test)
        hp = load_hyperparameters(path)
        model = build_sklearn_classifier_model(hp)
        model.fit(x_train_model, y_train)
        x_train_out.append(model.predict_proba(x_train_model))
        x_test_out.append(model.predict_proba(x_test_model))

    x_train_out = np.concatenate(x_train_out, axis=-1)
    x_test_out = np.concatenate(x_test_out, axis=-1)
    return x_train_out, x_test_out


def build_sklearn_classifier_model(hps: HyperParameters, preprocessing=True, tweet_level=False):
    """ Build an SkLearn classifier """
    sklearn_model = hps.Choice(
        "sklearn_model",
        # [LogisticRegression.__name__, SVC.__name__, RandomForestClassifier.__name__, XGBClassifier.__name__]
        [LogisticRegression.__name__, SVC.__name__, RandomForestClassifier.__name__]
    )
    steps = [("scaler", StandardScaler())]

    if sklearn_model == LogisticRegression.__name__:
        with hps.conditional_scope("sklearn_model", LogisticRegression.__name__):
            estimator = LogisticRegression(
                C=hps.Float("LogisticRegression_C", 0.0001, 1000),
                solver=hps.Choice("LogisticRegression_solver", ["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
            )
    elif sklearn_model == SVC.__name__:
        with hps.conditional_scope("sklearn_model", SVC.__name__):
            estimator = SVC(
                C=hps.Float("SVC_C", 0.0001, 1000),
                kernel=hps.Choice("SVC_kernel", ["poly", "rbf", "sigmoid"]),
                probability=True,
            )
    elif sklearn_model == RandomForestClassifier.__name__:
        with hps.conditional_scope("sklearn_model", RandomForestClassifier.__name__):
            estimator = RandomForestClassifier(
                n_estimators=hps.Choice("RandomForestClassifier_n_estimators", [50, 100, 200, 300, 400]),
                criterion=hps.Choice("RandomForestClassifier_criterion", ["gini", "entropy"]),
                min_samples_split=hps.Int("RandomForestClassifier_min_samples_split", 2, 8),
                min_samples_leaf=hps.Int("RandomForestClassifier_min_samples_leaf", 2, 6),
                min_impurity_decrease=hps.Float("RandomForestClassifier_min_impurity_decrease", 0, 1),
            )
    elif sklearn_model == XGBClassifier.__name__:
        with hps.conditional_scope("sklearn_model", XGBClassifier.__name__):
            estimator = XGBClassifier(
                learning_rate=hps.Float("XGBClassifier_learning_rate", 0.01, 0.1),
                gamma=hps.Float("XGBClassifier_gamma", 3, 7),
                max_depth=hps.Int("XGBClassifier_max_depth", 3, 6),
                min_child_weight=hps.Int("XGBClassifier_min_child_weight", 3, 6),
                subsample=hps.Float("XGBClassifier_subsample", 0.6, 1),
                colsample_bytree=hps.Float("XGBClassifier_colsample_bytree", 0.4, 0.7),
                colsample_bylevel=hps.Float("XGBClassifier_colsample_bylevel", 0.8, 1),
                colsample_bynode=hps.Float("XGBClassifier_colsample_bynode", 0.2, 0.5),
                reg_lambda=hps.Float("XGBClassifier_reg_lambda", 0.2, 1),
                reg_alpha=hps.Float("XGBClassifier_reg_alpha", 0.1, 0.5),
            )
    else:
        raise RuntimeError("Invalid SkLearn model name")

    steps.append(("estimator", estimator))
    if sklearn_model != XGBClassifier.__name__:  # Gradient Boosting Classifier doesn't have multi-collinearity issues
        steps.insert(0, ("PCA", PCA()))

    return SklearnTunerPipeline(steps, tweet_level=tweet_level) if preprocessing else estimator


def build_nn_classifier_model(hps: HyperParameters) -> Model:
    """ Build a neural network classifier """
    data_len = hps.get("input_datapoint_len")

    inputs = Input((data_len,))
    prev_layer = inputs
    for i in range(hps.Int("num_relu_layers", 1, 4)):
        relu = Dense(
            hps.Choice("layer_1_size", [data_len // 2, data_len, data_len * 2]) // max(1, i * 2),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(hps.Float("relu_kernel_reg", 0.0001, 0.1)),
            bias_regularizer=tf.keras.regularizers.l2(hps.Float("relu_bias_reg", 0.0001, 0.1)),
            activity_regularizer=tf.keras.regularizers.l2(hps.Float("relu_activity_reg", 0.0001, 0.1)),
        )(prev_layer)
        batch = BatchNormalization()(relu)
        prev_layer = Dropout(hps.Float("dropout_rate", 0.2, 0.5))(batch)
    linear_last = Dense(
        1,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(hps.Float("linear_kernel_reg", 0.0001, 0.1)),
        bias_regularizer=tf.keras.regularizers.l2(hps.Float("linear_bias_reg", 0.0001, 0.1)),
        activity_regularizer=tf.keras.regularizers.l2(hps.Float("linear_activity_reg", 0.0001, 0.1)),
    )(prev_layer)

    model = Model(inputs, linear_last)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hps.Float("learning_rate", 1e-5, 0.1)),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )
    return model


def tune_readability_model(x_train, y_train, project_name, **kwargs):
    return tune_sklearn_model(x_train, y_train, "readability_" + project_name, readability_tweet_extractor(), **kwargs)


def tune_ner_model(x_train, y_train, project_name, **kwargs):
    return tune_sklearn_model(x_train, y_train, "ner_" + project_name, ner_tweet_extractor(), **kwargs)


def tune_sentiment_model(x_train, y_train, project_name, **kwargs):
    return tune_sklearn_model(x_train, y_train, "sentiment_" + project_name, sentiment_tweet_extractor(), **kwargs)


def tune_combined_statistical_models(x_train, y_train, project_name, tune_sklearn_models=True, **kwargs):
    x_train_readability = readability_tweet_extractor().transform(x_train)
    x_train_ner = ner_tweet_extractor().transform(x_train)
    x_train_sentiment = sentiment_tweet_extractor().transform(x_train)
    x_train = np.concatenate([x_train_readability, x_train_ner, x_train_sentiment], axis=1)

    if tune_sklearn_models:
        return tune_sklearn_model(x_train, y_train, "combined_" + project_name, combined_tweet_extractor(), **kwargs)
    else:
        return tune_nn_model(x_train, y_train, "combined_nn_" + project_name, combined_tweet_extractor(), **kwargs)


def tune_combined_ensemble_model(x_train, y_train, project_name, **kwargs):
    extractors = [readability_tweet_extractor(), ner_tweet_extractor(), sentiment_tweet_extractor()]
    model_trial_paths = [
        f"../../training/statistical/readability_5/trial_5b6dd815ed192d976dbf5f9e4b24ce31/trial.json",
        f"../../training/statistical/ner_5/trial_5a645a5ce486c6f577c97060923bbb6b/trial.json",
        f"../../training/statistical/sentiment_5/trial_c0ebd26758c897f69f8aab7d4eccc427/trial.json",
    ]
    tuner = sklearn_classifier(
        "combined_ensemble_" + project_name,
        directory="../../training/statistical",
        hypermodel=partial(build_sklearn_classifier_model, preprocessing=False),
        **kwargs,
    )
    tuner.fit_data(
        x_train, y_train,
        partial(_stats_wrapper, model_extractors_and_paths=list(zip(extractors, model_trial_paths)))
    )
    tuner.search(x_train, y_train)
    return tuner


def tune_tweet_level_model(x_train, y_train, project_name, **kwargs):
    # Tweet-level tuning
    extractor = tweet_level_extractor()
    y_tl = np.asarray([v for v in y_train for _ in range(len(x_train[0]))])
    x_tl = extractor.transform(np.asarray([tweet for tweet_feed in x_train for tweet in tweet_feed]))
    tl_tuner = tune_sklearn_model(x_tl, y_tl, "tweet_level_" + project_name, **kwargs)

    tl_pipeline = build_sklearn_classifier_model(tl_tuner.get_best_hyperparameters(1)[0])
    tl_steps = tl_pipeline.steps
    tl_steps[-1] = (tl_steps[-1][0], PipelineEstimatorWrapper(tl_steps[-1][1]))
    tl_pipeline = SklearnTunerPipeline(tl_steps, tweet_level=True)

    # Pick best tweet-level model and find the best ensemble model for the results
    y_ul = y_train
    x_ul = x_tl.reshape((len(x_train), len(x_train[0]), len(x_tl[0])))

    def build_model(hp: HyperParameters):
        return SklearnTunerPipeline([
            ("tweet_level_pipeline", tl_pipeline),
            ("estimator", build_sklearn_classifier_model(hp, preprocessing=False)),
        ], tweet_level=False)
    ul_tuner = tune_sklearn_model(x_ul, y_ul, "tweet_level_ensemble_" + project_name, build_model=build_model, **kwargs)

    return tl_tuner, ul_tuner


def tune_nn_model(x_train, y_train, project_name, feature_extractor=None, tf_train_device="/gpu:0", **kwargs):
    # Extract features
    if feature_extractor is not None:
        x_train = feature_extractor.transform(x_train)

    # Setup Keras Tuner
    with tf.device(tf_train_device):
        hps = HyperParameters()
        hps.Choice("batch_size", [64, 96, 128, 160])
        hps.Fixed("epochs", 20)
        hps.Fixed("input_datapoint_len", x_train.shape[1])
        hps.Fixed("sklearn_model", "KerasFFNN")
        tuner = nn_tuner(project_name, hyperparameters=hps, **kwargs)
        tuner.search(
            x=x_train,
            y=y_train,
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=2),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs")
            ]
        )
        return tuner


def nn_tuner(project_name, hyperparameters=None, max_trials=30, directory="../../training/statistical"):
    return BayesianOptimizationCVTunerWithFitHyperParameters(
        hyperparameters=hyperparameters,
        hypermodel=build_nn_classifier_model,
        objective="val_loss",
        max_trials=max_trials,
        directory=directory,
        project_name=project_name,
    )


def tune_sklearn_model(x_train, y_train, project_name, feature_extractor=None,
                       build_model=build_sklearn_classifier_model, **kwargs):
    # Extract features
    if feature_extractor is not None:
        x_train = feature_extractor.transform(x_train)

    # Setup Keras Tuner
    tuner = sklearn_tuner(project_name, build_model=build_model, **kwargs)
    tuner.search(x_train, y_train)
    return tuner


def sklearn_tuner(project_name, build_model=build_sklearn_classifier_model, max_trials=30,
                  directory="../../training/statistical",
                  kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=3)):
    return Sklearn(
        oracle=BayesianOptimization(
            objective=Objective("score", "min"),  # minimise log loss
            max_trials=max_trials,
        ),
        hypermodel=build_model,
        scoring=make_scorer(log_loss, needs_proba=True),
        metrics=[accuracy_score, f1_score],
        cv=kfold,
        directory=directory,
        project_name=project_name,
    )
