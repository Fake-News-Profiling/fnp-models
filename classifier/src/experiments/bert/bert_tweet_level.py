import sys

import tensorflow as tf

from bert import BertIndividualTweetTokenizer, BertTweetFeedTokenizer
from experiments.experiment import AbstractBertExperiment
from experiments.handler import ExperimentHandler
from experiments.models import CompileOnFitKerasModel

"""
In this module, we experiment fine-tuning BERT on individual tweets or chunks of individual tweets.
"""


class BertTweetLevelExperiment(AbstractBertExperiment):
    """
    Train a BERT model on individual tweets, or chunks of individual tweets:
    * Bert input for "Bert.type" == "feed": "<user_i_tweet_i>. <user_i_tweet_i+1>. ...", <user_i_label>
      (up to max length of "Bert.hidden_size")
    * Bert input for "Bert.type" == "individual": "<user_i_tweet_i>", <user_i_label>
    """

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp)

        # Classifier layer
        dense_out = self.single_dense_layer(
            bert_output["pooled_output"],
            dropout_rate=hp.Choice("Bert.dropout_rate", [0., 0.1, 0.2]),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            no_l2_reg=True,
        )
        return CompileOnFitKerasModel(bert_input, dense_out, optimizer_learning_rate=hp.get("learning_rate"))

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return cls.tokenize_data(
            *cls.preprocess_data(hp, x_train, y_train, x_test, y_test),
            tokenizer_class=BertTweetFeedTokenizer if hp.get("Bert.type") == "feed" else BertIndividualTweetTokenizer,
            shuffle=True,
        )


class BertTweetLevelFfnnExperiment(AbstractBertExperiment):
    """
    Train a BERT model (or freeze it) on individual tweets, or chunks of individual tweets
    (same as BertTweetLevelExperiment), using a feed-forward neural network to classify the BERT embeddings
    """

    def build_model(self, hp, *args, **kwargs):
        # Get BERT inputs and outputs
        bert_trainable = hp.get("Bert.trainable")
        bert_input, bert_output = self.get_bert_layers(hp, trainable=bert_trainable)

        # FFNN layers
        hidden_size = hp.get("Bert.hidden_size")
        ffnn_input_size = hp.Choice("Bert.ffnn_input_size", [hidden_size, 3 * hidden_size // 4, hidden_size // 2])

        prev_layer = bert_output["pooled_output"]
        for i in range(hp.Int("Bert.num_ffnn_layers", 0, 3)):
            prev_layer = self.single_dense_layer(
                prev_layer,
                dense_units=ffnn_input_size // max(1, i * 2),
                dropout_rate=hp.Float("Bert.dropout_rate", 0, 0.5),
                dense_activation=hp.Fixed("Bert.dense_mid_layer_activation", "relu"),
                dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0., 0.0001, 0.001, 0.01]),
                dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0., 0.0001, 0.001, 0.01]),
                dense_activity_reg=0,
            )

        # Final classifier layer
        dense_out = self.single_dense_layer(
            prev_layer,
            dropout_rate=hp.Fixed("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Choice("Bert.dense_activation", ["relu", "linear"]),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0., 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0., 0.0001, 0.001, 0.01]),
            dense_activity_reg=0,
        )

        learning_rate = hp.get("learning_rate")
        if bert_trainable or ("optimizer" in hp and hp.get("optimizer") == "adamw"):
            # Use AdamW
            return CompileOnFitKerasModel(bert_input, dense_out, optimizer_learning_rate=learning_rate)
        else:
            # Use Adam
            return CompileOnFitKerasModel(
                bert_input, dense_out, selected_optimizer=tf.optimizers.Adam(learning_rate=learning_rate))

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

    preprocessing_choices = [
        # "[remove_emojis]",
        # "[remove_emojis, remove_punctuation]",
        "[remove_emojis, remove_tags]",
        "[remove_emojis, remove_tags, remove_punctuation]",
        # "[tag_emojis]",
        # "[tag_emojis, remove_punctuation]",
        # "[replace_emojis]",
        # "[replace_emojis_no_sep]",
        "[replace_emojis_no_sep, remove_tags]",
        "[replace_emojis_no_sep, remove_tags, remove_punctuation]",
        # "none",
    ]
    experiments = [  # TODO - Run to find best regularisation for linear layer
        (
            # Bert (128) Individual tweet-level with varied preprocessing functions
            BertTweetLevelExperiment,
            {
                "experiment_dir": "../training/bert_clf/tweet_level",
                "experiment_name": "indiv_2",
                "max_trials": 40,
                "hyperparameters": {
                    "epochs": 6,
                    "batch_size": [16, 32, 64, 80],
                    "learning_rate": [2e-5, 5e-5],
                    "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
                    "Bert.hidden_size": 128,
                    "Bert.preprocessing": preprocessing_choices,
                    "Bert.type": "individual",
                },
            }
            # ), (
            #     # Bert (128) Individual tweet-level with varied preprocessing functions
            #     BertTweetLevelFfnnExperiment,
            #     {
            #         "experiment_dir": "../training/bert_clf/tweet_level_ffnn",
            #         "experiment_name": "indiv_1",
            #         "max_trials": 50,
            #         "hyperparameters": {
            #             "epochs": 6,
            #             "batch_size": [16, 32, 64, 80],
            #             "learning_rate": [2e-5, 5e-5],
            #             "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"
            #             ,
            #             "Bert.hidden_size": 128,
            #             "Bert.preprocessing": preprocessing_choices,
            #             "Bert.type": "individual",
            #             "Bert.trainable": False,
            #         },
            #     }
        )
    ]
    with tf.device("/gpu:0"):
        handler = ExperimentHandler(experiments)
        handler.run_experiments(dataset_dir)

"""
Current best:
* bert_ffnn_4 - trial_be63599101309712ae1b659ba40808e8:
    * batch_size=32, dropout=0.3, activation=relu
    * model: BERT (128 out) -> drop -> batch -> dense (64 units) -> drop -> batch -> dense (1 unit)
* bert_ffnn_5 - trial_94a105c7810592295908dc29393e59ab:
    * batch_size=32, dropout=0.35, activation=relu
    * model: BERT (128 out) -> drop -> batch -> dense (1 unit)
* bert_ffnn_5 - trial_35307facb2039989941c7f9f875e6028:
    * batch_size=32, activation=relu
    * model: BERT (128 out) -> dense (1 unit)
* bert_ffnn_6 - trial_0b51ec0b3a25404c4fb4ee3bdeaa8d66:
    * batch_size=64, dropout=0.3, activation=relu
    * model: BERT (128 out) -> drop -> batch -> dense (1 unit)
* bert_ffnn_9 - trial_2481cabc193f7083ddce8cf11025bc8d: (reached 0.6 val_loss and 0.8 val_accuracy)
    * batch_size=24, dropout=0.22587, activation-relu
    * model: BERT (256 out) -> drop -> batch -> dense (256 unit) -> drop -> batch -> dense (128 unit) -> 
             drop -> batch -> dense (1 unit) -> 
* bert_ffnn_10 - trial_df5e725b3223e721f73bbc25a06897b4 (reached 0.6 val_loss and 0.79 val_accuracy)
    * batch_size=16, dropout=0.25470, activation=relu, lr=5e-5
    * model: BERT (256 out) -> drop -> batch -> dense (128 unit) -> drop -> batch -> dense (64 unit) -> 
             drop -> batch -> dense (1 unit) -> 

All of the above have 'feed_data_overlap' = 50 (default)
"""
