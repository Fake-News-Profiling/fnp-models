import os
from abc import ABC, abstractmethod
from typing import Any, Union

import tensorflow as tf
from dataclasses import dataclass, field
from kerastuner import HyperParameters, Objective
from kerastuner.tuners.bayesian import BayesianOptimizationOracle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fnpmodels.models.bert.models import bert_layers, bert_tweet_preprocessor
from fnpmodels.models.bert.tokenize import BertTweetFeedTokenizer, tokenize_bert_input
from fnpmodels.processing.statistical.extractor import TweetStatsExtractor
from . import allow_gpu_memory_growth, load_hyperparameters
from .tuners import BayesianOptimizationCV, SklearnCV, kfold_split_wrapper

allow_gpu_memory_growth()


@dataclass
class ExperimentConfig:
    experiment_dir: str
    experiment_name: str
    hyperparameters: dict = field(default_factory=dict)
    max_trials: int = 30
    num_cv_splits: int = 5


class AbstractExperiment(ABC):
    """ Abstract base class for conducting a Keras Tuner tuning experiment """

    def __init__(self, config: ExperimentConfig):
        self.experiment_directory = config.experiment_dir
        self.experiment_name = config.experiment_name
        self.config = config
        self.hyperparameters = self.parse_to_hyperparameters(self.config.hyperparameters)
        self.tuner = None

    @abstractmethod
    def build_model(self, hp):
        """ Build the Keras Tuner model for this experiment """
        pass

    @abstractmethod
    def run(self, x, y, *args, **kwargs):
        """ Run this experiment """
        pass

    @classmethod
    def parse_to_hyperparameters(cls, hyperparameters: Union[str, dict]):
        """
        Parse a filepath to a saved KerasTuner HyperParameters config file, or if a dict is passed then parse it to a
        HyperParameters object. The dict should be of the form:
        {"hp_name": {"type": "hp_type", "args": [...], "kwargs": {...}}, ...}

        For simplicity:
        * `hp.Fixed` values can also be set as: {"hp_name": fixed_hp_value, ...}
        * `hp.Choice` values can also be set as: {"hp_name": [hp_choice_1, ...], ...}
        """
        if hyperparameters is None:
            return None
        if isinstance(hyperparameters, str):
            return load_hyperparameters(hyperparameters)

        hp = HyperParameters()
        # Add each hyperparameter to `hp`
        for name, value in hyperparameters.items():
            if isinstance(value, dict) and "condition" in value:
                with hp.conditional_scope(value["condition"]["name"], value["condition"]["value"]):
                    if "value" in value:
                        value = value["value"]

                    cls._make_hp(hp, name, value)
            else:
                cls._make_hp(hp, name, value)

        return hp

    @staticmethod
    def _make_hp(hp: HyperParameters, name: str, value: Any):
        if isinstance(value, dict):
            hp_type = value["type"]
            hp_type_args = value.get(["args"], [])
            hp_type_kwargs = value.get(["kwargs"], {})

            hyperparameter = {
                "int": hp.Int,
                "float": hp.Float,
                "bool": hp.Boolean,
                "choice": hp.Choice,
                "fixed": hp.Fixed,
            }[hp_type]
            hyperparameter(name, *hp_type_args, **hp_type_kwargs)
        elif isinstance(value, list):
            hp.Choice(name, value)
        else:
            hp.Fixed(name, value)


class AbstractSklearnExperiment(AbstractExperiment, ABC):
    """
    Abstract base class for conducting a Keras Tuner tuning experiment with an Sklearn model.
    Each Sklearn model is tuned individually, so a list of sklearn model types must be passed in the config
    hyperparameters.
    """

    def __init__(self, config: ExperimentConfig, tuner_initial_points: int = None):
        super().__init__(config)
        self.tuner_initial_points = tuner_initial_points or (config.max_trials // 5 if config.max_trials > 10 else None)

    def make_tuner(self, hp: HyperParameters, model_type: str) -> SklearnCV:
        return SklearnCV(
            oracle=BayesianOptimizationOracle(
                objective=Objective("score", "min"),  # minimise cross-entropy loss
                max_trials=self.config.max_trials,
                hyperparameters=hp,
                num_initial_points=self.tuner_initial_points,
            ),
            hypermodel=self.build_model,
            scoring=make_scorer(tf.keras.losses.binary_crossentropy),
            metrics=[accuracy_score, f1_score],
            cv=StratifiedKFold(n_splits=self.config.num_cv_splits, shuffle=True, random_state=1),
            directory=os.path.join(self.experiment_directory, self.experiment_name),
            project_name=model_type,
        )

    def run(self, x, y, callbacks=None, *args, **kwargs):
        # Processing the initial data is computationally costly, so do it once before we hyperparameter tune
        if hasattr(self, "input_data_transformer") and callable(self.input_data_transformer):
            x = self.input_data_transformer(x)

        cv_data = None
        if hasattr(self, "cv_data_transformer") and callable(self.cv_data_transformer):
            cv = StratifiedKFold(n_splits=self.config.num_cv_splits, shuffle=True, random_state=1)
            cv_data = [self.cv_data_transformer(self.hyperparameters, *cv_fold)
                       for cv_fold in kfold_split_wrapper(cv, x, y)]

        # Run a search for each model type
        model_types = self.config.hyperparameters.setdefault(
            "Sklearn.model_type", ["LogisticRegression", "SVC", "RandomForestClassifier", "XGBClassifier"])
        if isinstance(model_types, str):
            model_types = [model_types]

        for model_type in model_types:
            # Build a HyperParameters object with just this model_type
            hp_dict = self.config.hyperparameters.copy()
            hp_dict["Sklearn.model_type"] = model_type
            hp = self.parse_to_hyperparameters(hp_dict)

            tuner = self.make_tuner(hp, model_type)
            tuner.cv_data = cv_data
            print("Performing search for model_type:", model_type)
            tuner.search(x, y)

    @classmethod
    def build_model(cls, hp):
        """ Build an Sklearn pipeline with a final estimator """
        estimator = cls.select_sklearn_model(hp)
        steps = [("scaler", StandardScaler()), ("estimator", estimator)]

        # Add PCA to deal with multi-collinearity issues in data (Gradient Boosting doesn't have this issue)
        if hp.Fixed("Sklearn.use_pca", True) and hp.get("Sklearn.model_type") != XGBClassifier.__name__:
            steps.insert(0, ("PCA", PCA()))

        return Pipeline(steps)

    @staticmethod
    def select_sklearn_model(hp):
        """ Select and instantiate an Sklearn estimator """
        model_type = hp.get("Sklearn.model_type")
        if model_type == LogisticRegression.__name__:
            with hp.conditional_scope("Sklearn.model_type", LogisticRegression.__name__):
                # Use a log distribution as it's more likely we'll want less regularisation
                estimator = LogisticRegression(
                    C=hp.Float("Sklearn.LogisticRegression.C", 0.01, 500, sampling="log"),
                )
        elif model_type == SVC.__name__:
            with hp.conditional_scope("Sklearn.model_type", SVC.__name__):
                estimator = SVC(
                    C=hp.Float("Sklearn.SVC.C", 0.01, 500, sampling="log"),
                    probability=True,
                )
        elif model_type == RandomForestClassifier.__name__:
            with hp.conditional_scope("Sklearn.model_type", RandomForestClassifier.__name__):
                estimator = RandomForestClassifier(
                    n_estimators=hp.Int("Sklearn.RandomForestClassifier.n_estimators", 1, 500),
                    criterion=hp.Choice("Sklearn.RandomForestClassifier.criterion", ["gini", "entropy"]),
                    min_samples_split=hp.Int("Sklearn.RandomForestClassifier.min_samples_split", 2, 8),
                    min_samples_leaf=hp.Int("Sklearn.RandomForestClassifier.min_samples_leaf", 2, 8),
                    min_impurity_decrease=hp.Float("Sklearn.RandomForestClassifier.min_impurity_decrease", 0, 1),
                )
        elif model_type == XGBClassifier.__name__:
            with hp.conditional_scope("Sklearn.model_type", XGBClassifier.__name__):
                estimator = XGBClassifier(
                    verbosity=0,
                    learning_rate=hp.Float("Sklearn.XGBClassifier.learning_rate", 0.01, 0.2),
                    gamma=hp.Float("Sklearn.XGBClassifier.gamma", 0, 8),
                    max_depth=hp.Int("Sklearn.XGBClassifier.max_depth", 2, 8),
                    min_child_weight=hp.Int("Sklearn.XGBClassifier.min_child_weight", 1, 8),
                    subsample=hp.Float("Sklearn.XGBClassifier.subsample", 0.5, 1),
                    colsample_bytree=hp.Float("Sklearn.XGBClassifier.colsample_bytree", 0.2, 1),
                    colsample_bylevel=hp.Float("Sklearn.XGBClassifier.colsample_bylevel", 0.5, 1),
                    colsample_bynode=hp.Float("Sklearn.XGBClassifier.colsample_bynode", 0.2, 0.8),
                    reg_lambda=hp.Float("Sklearn.XGBClassifier.reg_lambda", 0., 1),
                )
        else:
            raise RuntimeError("Invalid Sklearn model type:", model_type)

        return estimator

    @staticmethod
    def get_extractor(*args, **kwargs) -> TweetStatsExtractor:
        pass


class AbstractTfExperiment(AbstractExperiment, ABC):
    """ Abstract base class for conducting a Keras Tuner tuning experiment with a TensorFlow model """

    def __init__(self, config: ExperimentConfig, tuner_class=BayesianOptimizationCV):
        super().__init__(config)

        self.tuner = tuner_class(
            n_splits=self.config.num_cv_splits,
            preprocess=self.preprocess_cv_data,
            hypermodel=self.build_model,
            hyperparameters=self.hyperparameters,
            objective="val_loss",
            max_trials=self.config.max_trials,
            directory=self.experiment_directory,
            project_name=self.experiment_name,
        )

    def run(self, x, y, callbacks=None, *args, **kwargs):
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.TerminateOnNaN(),
                # tf.keras.callbacks.EarlyStopping("val_loss", patience=2),
                tf.keras.callbacks.TensorBoard(log_dir=self.experiment_directory + "/" + self.experiment_name + "/logs")
            ]
        if hasattr(self, "cv_data_transformer"):
            self.tuner.fit_data(x, y, self.cv_data_transformer)

        self.tuner.search(x=x, y=y, callbacks=callbacks, verbose=2, *args, **kwargs)

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        """ Preprocessing function applied to every Cross-Validation data split, before data is fed into the model """
        return hp, x_train, y_train, x_test, y_test


class AbstractBertExperiment(AbstractTfExperiment, ABC):
    """ Abstract base class for conducting a Keras Tuner tuning experiment with a BERT-base TensorFlow model """

    @staticmethod
    def single_dense_layer(inputs, dropout_rate, dense_activation,
                           no_l2_reg=False, dense_kernel_reg=None, dense_units=1):
        dropout = tf.keras.layers.Dropout(dropout_rate)(inputs)
        batch = tf.keras.layers.BatchNormalization()(dropout)
        if no_l2_reg:
            dense_out = tf.keras.layers.Dense(
                units=dense_units,
                activation=dense_activation,
            )(batch)
        else:
            dense_out = tf.keras.layers.Dense(
                units=dense_units,
                activation=dense_activation,
                kernel_regularizer=tf.keras.regularizers.l2(dense_kernel_reg),
            )(batch)
        return dense_out

    @staticmethod
    def get_bert_layers(hp, trainable=True, **kwargs):
        return bert_layers(
            hp.get("Bert.encoder_url"),
            trainable=trainable,
            hidden_layer_size=hp.get("Bert.hidden_size"),
            **kwargs,
        )

    @staticmethod
    def tokenize_data(hp, x_train, y_train, x_test, y_test, tokenizer_class, **kwargs):
        tokenized = tokenize_bert_input(
            encoder_url=hp.get("Bert.encoder_url"),
            hidden_layer_size=hp.get("Bert.hidden_size"),
            tokenizer_class=tokenizer_class,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            feed_overlap=hp.get("Bert.feed_data_overlap") if isinstance(tokenizer_class, BertTweetFeedTokenizer) else 0,
            **kwargs,
        )
        return (hp, *tokenized)

    @staticmethod
    def preprocess_data(hp, x_train, y_train, x_test, y_test):
        preprocessor = bert_tweet_preprocessor(hp)
        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)
        return hp, x_train, y_train, x_test, y_test
