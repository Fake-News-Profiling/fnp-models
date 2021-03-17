import sys

import numpy as np
import tensorflow as tf

from bert import BertIndividualTweetTokenizer, BertTweetFeedTokenizer
from data import parse_dataset
from experiments.experiment import AbstractBertExperiment


class BertPooledOutputExperiment(AbstractBertExperiment):
    """
    Train a BERT model, using different methods to produce the pooled_output of BERT
    """

    def build_model(self, hp):
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
            dense_pooled = tf.keras.layers.Dense(hp.get("bert_size"), activation="tanh")(pooled_outputs)

        # Classifier layer
        dense_out = self.single_dense_layer(
            dense_pooled,
            dropout_rate=hp.Float("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0., 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0., 0.0001, 0.001, 0.01]),
            dense_activity_reg=hp.Choice("Bert.dense_activity_reg", [0., 0.0001, 0.001, 0.01]),
        )

        return self.compile_model_with_adamw(hp, bert_input, dense_out)

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
    experiment_dir = sys.argv[2]
    x, y = parse_dataset(dataset_dir, "en")

    # BertPooledOutputExperiment
    config = {
        "max_trials": 20,
        "hyperparameters": {
            "epochs": 8,
            "batch_size": 64,
            "learning_rate": [2e-5, 5e-5],
            "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "Bert.hidden_size": 128,
            "Bert.preprocessing": "[remove_emojis, remove_tags]",
        }
    }
    experiment = BertPooledOutputExperiment(experiment_dir, "pooled_output_1", config)
    experiment.run(x, y)
