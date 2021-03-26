import sys

import tensorflow as tf

from bert import BertIndividualTweetTokenizer, BertTweetFeedTokenizer
from bert.models import extract_bert_pooled_output
from experiments.experiment import AbstractBertExperiment, ExperimentConfig
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel
from experiments.tuners import GridSearchCV


class BertPooledOutputExperiment(AbstractBertExperiment):
    """
    Train a BERT model, using different methods to produce the pooled_output of BERT
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, tuner_class=GridSearchCV)

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)

        # `bert_output["encoder_outputs"]` is a list of tensors of outputs of each encoder in the BERT encoder stack.
        # First we select which encoder outputs to use, and then apply a pooling strategy to them.
        # encoder_outputs.shape == (num_encoders, batch_size, seq_len, hidden_size)
        selected_encoder_outputs = hp.get("selected_encoder_outputs")
        pooling_layer = tf.keras.layers.Dense(hp.get("Bert.hidden_size"), activation="tanh")
        pooled_output = extract_bert_pooled_output(bert_output, pooling_layer, selected_encoder_outputs, )

        # Classifier layer
        dense_out = self.single_dense_layer(
            pooled_output,
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
    ]
    encoder_output_choices = [
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
    ]

    experiments = [
        (
            # Bert (128) Individual with different pooled output methods
            BertPooledOutputExperiment,
            {
                "experiment_dir": "../training/bert_clf/pooled_output",
                "experiment_name": "indiv_1",
                "max_trials": 50,
                "hyperparameters": {
                    "epochs": 8,
                    "batch_size": [32, 80, 128],
                    "learning_rate": [2e-5, 5e-5],
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": ["[remove_emojis, remove_tags]",
                                           "[remove_emojis, remove_tags, remove_punctuation]",
                                           "[replace_emojis_no_sep, remove_tags]"],
                    "Bert.type": "individual",
                    "selected_encoder_outputs": ["default", "2nd_to_last_hidden_layer"],
                    "Bert.dropout_rate": 0.1,
                    "Bert.dense_kernel_reg": 0.,
                    "Bert.dense_bias_reg": 0.01
                },
            }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        # handler.run_experiments(dataset_dir)
        handler.plot_experiment(
            0,
            trial_label_generator=lambda t, hp: f"{hp.get('selected_encoder_outputs')}",
            trial_aggregator=lambda hp: f"{hp.get('selected_encoder_outputs')}",
        )
        # handler.plot_experiment(
        #     0,
        #     trial_label_generator=lambda t, hp: f"{hp.get('Bert.preprocessing')}",
        #     trial_aggregator=lambda hp: f"{hp.get('Bert.preprocessing')}",
        # )
        # handler.plot_experiment(
        #     0,
        #     trial_label_generator=lambda t, hp: f"{hp.get('batch_size')}",
        #     trial_aggregator=lambda hp: f"{hp.get('batch_size')}",
        # )
        # handler.plot_experiment(
        #     0,
        #     trial_label_generator=lambda t, hp: f"{hp.get('Bert.dense_kernel_reg')}",
        #     trial_aggregator=lambda hp: f"{hp.get('Bert.dense_kernel_reg')}",
        # )
        # handler.plot_experiment(
        #     0,
        #     trial_label_generator=lambda t, hp: f"{hp.get('Bert.dense_bias_reg')}",
        #     trial_aggregator=lambda hp: f"{hp.get('Bert.dense_bias_reg')}",
        # )
        # handler.plot_experiment(
        #     0,
        #     trial_label_generator=lambda t, hp: f"{hp.get('learning_rate')}",
        #     trial_aggregator=lambda hp: f"{hp.get('learning_rate')}",
        # )
