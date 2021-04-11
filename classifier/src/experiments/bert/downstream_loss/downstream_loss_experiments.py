import sys

import tensorflow as tf

from experiments.bert.downstream_loss.bert_experiment_models import BertTrainedOnDownstreamLoss
from experiments.handler import ExperimentHandler


def downstream_loss_tuning_handler():
    default_hps = {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        "Bert.hidden_size": 128,
        "Bert.preprocessing": "[remove_emojis, remove_tags]",
        "selected_encoder_outputs": "default",
        "Bert.pooler": "max",
        "Bert.dropout_rate": 0.1,
        "Bert.dense_kernel_reg": 0.,
        "Bert.num_hidden_layers": 0,
        "Bert.use_batch_norm": False,
        "Bert.tweet_feed_len": 10,
    }
    experiments = [
        (
            # Preprocessing choice
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "preprocessing",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.preprocessing": [
                        "[remove_emojis, remove_tags]",
                        "[remove_emojis, remove_tags, remove_punctuation]",
                        "[replace_emojis_no_sep, remove_tags]",
                    ],
                },
            }
        ), (
            # BERT pooled_output strategy
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "pooled_output",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.pooler": "concat",
                    "selected_encoder_outputs": [
                        "default",
                        "2nd_to_last_hidden_layer",
                        "sum_all_hidden_layers",
                        "sum_last_4_hidden_layers",
                        "concat_last_4_hidden_layers",
                    ],
                },
            }
        ), (
            # BERT pooler
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "pooler",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.pooler": ["max", "average", "concat"],
                },
            }
        ), (
            # Dropout rate
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "dropout_rate",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.dropout_rate": [0., 0.1, 0.2, 0.3, 0.4],
                },
            }
        ), (
            # Dense classifier kernel regularisation
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "kernel_reg",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.dense_kernel_reg": [0., 0.00001, 0.0001, 0.001, 0.01],
                },
            }
        ), (
            # Final dense activation
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "dense_activation",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.dense_activation": ["linear", "sigmoid", "tanh", "relu"],
                },
            }
        ), (
            # FFNN
            BertTrainedOnDownstreamLoss,
            {
                "experiment_dir": "../training/bert_clf/downstream_loss",
                "experiment_name": "ffnn_tanh",
                "max_trials": 36,
                "hyperparameters": {
                    **default_hps,
                    "Bert.num_hidden_layers": [0, 1, 2, 3, 4],
                    "Bert.hidden_dense_activation": "tanh",
                },
            }
        )
    ]
    return ExperimentHandler(experiments)


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    with tf.device("/gpu:0"):
        # Hyperparameter tuning experiments
        hp_handler = downstream_loss_tuning_handler()
        # hp_handler.run_experiments(dataset_dir)

        # Best models experiments
        model_experiments = [
            (
                BertTrainedOnDownstreamLoss,
                {
                    "experiment_dir": "../training/bert_clf/downstream_loss",
                    "experiment_name": "best_models",
                    "max_trials": 36,
                    "hyperparameters": {
                        "epochs": 5,
                        "batch_size": 8,
                        "learning_rate": 2e-5,
                        "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                        "Bert.hidden_size": 128,
                        "Bert.preprocessing": [
                            "[replace_emojis_no_sep, remove_tags]",
                            "[remove_emojis, remove_tags, remove_punctuation]",
                        ],
                        "selected_encoder_outputs": [
                            "default",
                            "concat_last_4_hidden_layers",
                        ],
                        "Bert.pooler": [
                            "max",
                            "concat"
                        ],
                        "Bert.dense_kernel_reg": 0.0001,
                        "Bert.use_batch_norm": False,
                        "Bert.num_hidden_layers": 0,
                        "Bert.dense_activation": [
                            "tanh",
                            "linear",
                        ],
                        "Bert.dropout_rate": 0.1,
                        "Bert.tweet_feed_len": 10,
                    },
                }
            ),
        ]
        model_handler = ExperimentHandler(model_experiments)
        # model_handler.run_experiments(dataset_dir)
