import sys

import tensorflow as tf

from bert import BertIndividualTweetTokenizer, BertTweetFeedTokenizer
from bert.models import extract_bert_pooled_output
from experiments.experiment import AbstractBertExperiment, ExperimentConfig
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel
from experiments.tuners import GridSearchCV


class BertIndividualExperiment(AbstractBertExperiment):
    """ Train a BERT model on individual tweets """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, tuner_class=GridSearchCV)
        self.tuner.num_folds = 3
        self.tuner.oracle.num_completed_trials = 0

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)

        # `bert_output["encoder_outputs"]` is a list of tensors of outputs of each encoder in the BERT encoder stack.
        # First we select which encoder outputs to use, and then apply a pooling strategy to them.
        # encoder_outputs.shape == (num_encoders, batch_size, seq_len, hidden_size)
        selected_encoder_outputs = hp.get("selected_encoder_outputs")
        pooling_layer = tf.keras.layers.Dense(hp.get("Bert.hidden_size"), activation="tanh")
        pooled_output = extract_bert_pooled_output(bert_output, pooling_layer, selected_encoder_outputs)

        # Classifier layer
        dense_out = self.single_dense_layer(
            pooled_output,
            dropout_rate=hp.Fixed("Bert.dropout_rate", 0.1),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.get("Bert.dense_kernel_reg"),
        )

        return CompileOnFitKerasModel(bert_input, dense_out, optimizer_learning_rate=hp.get("learning_rate"))

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return cls.tokenize_data(
            *cls.preprocess_data(hp, x_train, y_train, x_test, y_test),
            tokenizer_class=BertTweetFeedTokenizer if hp.get("Bert.type") == "feed" else BertIndividualTweetTokenizer,
            shuffle=True,
        )
