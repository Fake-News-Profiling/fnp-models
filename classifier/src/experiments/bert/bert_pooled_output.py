import sys

import numpy as np
import tensorflow as tf

from bert import BertIndividualTweetTokenizer, BertTweetFeedTokenizer
from experiments.experiment import AbstractBertExperiment, ExperimentConfig
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel
from experiments.tuners import GridSearchCV


class BertPooledOutputExperiment(AbstractBertExperiment):
    """
    Train a BERT model, using different methods to produce the pooled_output of BERT
    """

    def build_model(self, hp, *args, **kwargs):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)

        # `bert_output["encoder_outputs"]` is a list of tensors of outputs of each encoder in the BERT encoder stack.
        # First we select which encoder outputs to use, and then apply a pooling strategy to them.
        # encoder_outputs.shape == (num_encoders, batch_size, seq_len, hidden_size)
        encoder_outputs = bert_output["encoder_outputs"]
        selected_encoder_outputs = hp.Choice("selected_encoder_outputs", [
            "default",
            "first_layer",
            "2nd_to_last_hidden_layer",
            "last_hidden_layer",
            "sum_all_but_last_hidden_layers",
            "sum_all_hidden_layers",
            "sum_4_2nd_to_last_hidden_layers",
            "sum_last_4_hidden_layers",
            "concat_4_2nd_to_last_hidden_layers",
            "concat_last_4_hidden_layers",
        ])

        if selected_encoder_outputs == "default":
            dense_pooled = bert_output["pooled_output"]
        else:
            encoder_outputs = encoder_outputs[{
                "first_layer": np.s_[:1],
                "2nd_to_last_hidden_layer": np.s_[-2:-1],
                "last_hidden_layer": np.s_[-1:],
                "sum_all_but_last_hidden_layers": np.s_[:-1],
                "sum_all_hidden_layers": np.s_[:],
                "sum_4_2nd_to_last_hidden_layers": np.s_[-5:-1],
                "sum_last_4_hidden_layers": np.s_[:-4],
                "concat_4_2nd_to_last_hidden_layers": np.s_[-5:-1],
                "concat_last_4_hidden_layers": np.s_[:-4],
            }[selected_encoder_outputs]]

            # Pool `encoder_outputs` by summing or concatenating
            if selected_encoder_outputs.startswith("concat"):
                # Concatenate layer outputs, and extract the concatenated '[CLF]' embeddings
                # pooled_outputs.shape == (batch_size, len(encoder_outputs) * hidden_size)
                pooled_outputs = tf.concat(encoder_outputs, axis=-1)[:, 0, :]
            else:
                # Extract the '[CLF]' embeddings of each layer, and then sum them
                pooled_outputs = tf.convert_to_tensor(encoder_outputs)[:, :, 0, :]
                # pooled_outputs.shape == (batch_size, hidden_size)
                pooled_outputs = tf.reduce_sum(pooled_outputs, axis=0)

            # Pass pooled_outputs through a tanh layer (as they did in the original BERT paper)
            dense_pooled = tf.keras.layers.Dense(hp.get("Bert.hidden_size"), activation="tanh")(pooled_outputs)

        # Classifier layer
        dense_out = self.single_dense_layer(
            dense_pooled,
            dropout_rate=hp.Fixed("Bert.dropout_rate", 0.1),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Fixed("Bert.dense_kernel_reg", 0.),
            dense_bias_reg=hp.Fixed("Bert.dense_bias_reg", 0.),
            dense_activity_reg=0,
        )

        return CompileOnFitKerasModel(bert_input, dense_out, optimizer_learning_rate=hp.get("learning_rate"))

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return cls.tokenize_data(
            *cls.preprocess_data(hp, x_train, y_train, x_test, y_test),
            tokenizer_class=BertTweetFeedTokenizer if hp.get("Bert.type") == "feed" else BertIndividualTweetTokenizer,
            shuffle=True,
        )


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
            BertPooledOutputExperiment,
            {
                "experiment_dir": "../training/bert_clf/pooled_output",
                "experiment_name": "indiv_3",
                "max_trials": 50,
                "hyperparameters": {
                    "epochs": 4,
                    "batch_size": [32, 64, 80, 128],
                    "learning_rate": [2e-5, 5e-5],
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": preprocessing_choices,
                    "Bert.type": "individual",
                    "selected_encoder_outputs": ["default", "first_layer", "2nd_to_last_hidden_layer"],
                    "Bert.dropout_rate": [0., 0.1, 0.2],
                    "Bert.dense_kernel_reg": [0., 0.001, 0.01],
                    "Bert.dense_bias_reg": [0., 0.001, 0.01],
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        handler.run_experiments(dataset_dir)
        handler.plot_experiment(0, trial_label_generator=lambda t, hp: f"{hp.get('selected_encoder_outputs')}")
