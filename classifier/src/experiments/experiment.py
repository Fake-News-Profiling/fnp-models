from abc import ABC, abstractmethod
from functools import partial

import tensorflow as tf
from official.nlp.optimization import AdamWeightDecay
from kerastuner import HyperParameters

from bert import BertTweetFeedTokenizer
from bert.models import tokenize_bert_input, bert_layers
from experiments import allow_gpu_memory_growth
from experiments.tuners import BayesianOptimizationCV
import data.preprocess as pre


allow_gpu_memory_growth()


class AbstractExperiment(ABC):
    """ Abstract base class for conducting a Keras Tuner tuning experiment """

    def __init__(self, experiment_directory: str, experiment_name: str, config: dict):
        self.experiment_directory = experiment_directory
        self.experiment_name = experiment_name
        self.config = config
        self.hyperparameters = self.parse_to_hyperparameters(self.config["hyperparameters"])
        self.tuner = None

    @abstractmethod
    def build_model(self, hp):
        """ Build the Keras Tuner model for this experiment """
        pass

    @abstractmethod
    def run(self, x, y, *args, **kwargs):
        """ Run this experiment """
        pass

    @staticmethod
    def parse_to_hyperparameters(hyperparameters_dict: dict):
        """
        Parse a dict of hyperparameters to a KerasTuner HyperParameters object. The dict should be of the form:
        {"hp_name": {"type": "hp_type", "args": [...], "kwargs": {...}}, ...}

        For simplicity:
        * `hp.Fixed` values can also be set as: {"hp_name": fixed_hp_value, ...}
        * `hp.Choice` values can also be set as: {"hp_name": [hp_choice_1, ...], ...}
        """
        if hyperparameters_dict is None:
            return None

        hp = HyperParameters()
        # Add each hyperparameter to `hp`
        for name, value in hyperparameters_dict.items():
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

        return hp


class AbstractTfExperiment(AbstractExperiment, ABC):
    """ Abstract base class for conducting a Keras Tuner tuning experiment with a TensorFlow model """

    def __init__(self, experiment_directory: str, experiment_name: str, config: dict):
        super().__init__(experiment_directory, experiment_name, config)
        self._init_tuner(config.get("max_trials", 30))

    def _init_tuner(self, max_trials):
        self.tuner = BayesianOptimizationCV(
            preprocess=self.preprocess_cv_data,
            hypermodel=self.build_model,
            hyperparameters=self.hyperparameters,
            objective="val_loss",
            max_trials=max_trials,
            directory=self.experiment_directory,
            project_name=self.experiment_name,
        )

    def run(self, x, y, callbacks=None, *args, **kwargs):
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.EarlyStopping("val_loss", patience=2),
                tf.keras.callbacks.TensorBoard(log_dir=self.experiment_directory + "/" + self.experiment_name + "/logs")
            ]
        if hasattr(self, "cv_data_transformer"):
            self.tuner.fit_data(x, y, self.cv_data_transformer)

        self.tuner.search(x=x, y=y, callbacks=callbacks, *args, **kwargs)

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        """ Preprocessing function applied to every Cross-Validation data split, before data is fed into the model """
        return hp, x_train, y_train, x_test, y_test


class AbstractBertExperiment(AbstractTfExperiment, ABC):
    """ Abstract base class for conducting a Keras Tuner tuning experiment with a BERT-base TensorFlow model """

    @staticmethod
    def compile_model_with_adamw(hp, model_inputs, model_outputs, learning_rate=None):
        """ Compile a model using AdamWeightDecay """
        if learning_rate is None:
            learning_rate = hp.get("learning_rate")

        model = tf.keras.Model(model_inputs, model_outputs)
        model.compile(
            optimizer=AdamWeightDecay(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=tf.metrics.BinaryAccuracy(),
        )
        return model

    @staticmethod
    def single_dense_layer(inputs, dropout_rate, dense_activation, dense_kernel_reg,
                           dense_bias_reg, dense_activity_reg, dense_units=1):
        dropout = tf.keras.layers.Dropout(dropout_rate)(inputs)
        batch = tf.keras.layers.BatchNormalization()(dropout)
        dense_out = tf.keras.layers.Dense(
            units=dense_units,
            activation=dense_activation,
            kernel_regularizer=tf.keras.regularizers.l2(dense_kernel_reg),
            bias_regularizer=tf.keras.regularizers.l2(dense_bias_reg),
            activity_regularizer=tf.keras.regularizers.l2(dense_activity_reg),
        )(batch)
        return dense_out

    @staticmethod
    def get_bert_layers(hp, trainable=True):
        return bert_layers(
            hp.get("Bert.encoder_url"),
            trainable=trainable,
            hidden_layer_size=hp.get("Bert.hidden_size")
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
        data_transformers = {
            "[remove_emojis]": [pre.remove_emojis, pre.replace_unicode, pre.replace_tags],
            "[remove_emojis, remove_punctuation]": [
                pre.remove_emojis, pre.replace_unicode, pre.remove_colons, pre.remove_punctuation_and_non_printables,
                pre.replace_tags, pre.remove_hashtags],
            "[remove_emojis, remove_tags]": [
                pre.remove_emojis, pre.replace_unicode, partial(pre.replace_tags, remove=True)],
            "[remove_emojis, remove_tags, remove_punctuation]": [
                pre.remove_emojis, pre.replace_unicode, pre.remove_colons, pre.remove_punctuation_and_non_printables,
                partial(pre.replace_tags, remove=True), pre.remove_hashtags],
            "[tag_emojis]": [partial(pre.replace_emojis, with_desc=False), pre.replace_unicode, pre.replace_tags],
            "[tag_emojis, remove_punctuation]": [
                partial(pre.replace_emojis, with_desc=False), pre.replace_unicode, pre.remove_colons,
                pre.remove_punctuation_and_non_printables, pre.replace_tags, pre.remove_hashtags],
            "[replace_emojis]": [pre.replace_emojis, pre.replace_unicode, pre.replace_tags],
            "[replace_emojis_no_sep]": [partial(pre.replace_emojis, sep=""), pre.replace_unicode, pre.replace_tags],
            "[replace_emojis_no_sep, remove_tags]": [
                partial(pre.replace_emojis, sep=""), pre.replace_unicode, partial(pre.replace_tags, remove=True)],
            "[replace_emojis_no_sep, remove_tags, remove_punctuation]": [
                partial(pre.replace_emojis, sep=""), pre.replace_unicode, pre.remove_colons,
                pre.remove_punctuation_and_non_printables, partial(pre.replace_tags, remove=True), pre.remove_hashtags],
        }[hp.get("Bert.preprocessing")]  # TODO - tag numbers?

        # Preprocess data
        preprocessor = pre.BertTweetPreprocessor(
            [pre.tag_indicators, pre.replace_xml_and_html] + data_transformers + [pre.remove_extra_spacing])
        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)

        return hp, x_train, y_train, x_test, y_test


