import sys

from bert import BertIndividualTweetTokenizer, BertTweetFeedTokenizer
from data import parse_dataset
from experiments.experiment import AbstractBertExperiment


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
            dropout_rate=hp.Float("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0., 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0., 0.0001, 0.001, 0.01]),
            dense_activity_reg=hp.Fixed("Bert.dense_activity_reg", 0),
        )

        return self.compile_model_with_adamw(hp, bert_input, dense_out)

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

    def build_model(self, hp):
        # Get BERT inputs and outputs
        bert_input, bert_output = self.get_bert_layers(hp, trainable=hp.get("Bert.trainable"))

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
                dense_activity_reg=None,
            )

        # Final classifier layer
        dense_out = self.single_dense_layer(
            prev_layer,
            dropout_rate=hp.Float("Bert.dropout_rate", 0, 0.5),
            dense_activation=hp.Fixed("Bert.dense_activation", "linear"),
            dense_kernel_reg=hp.Choice("Bert.dense_kernel_reg", [0., 0.0001, 0.001, 0.01]),
            dense_bias_reg=hp.Choice("Bert.dense_bias_reg", [0., 0.0001, 0.001, 0.01]),
            dense_activity_reg=None,
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

    preprocessing_choices = [
        "[remove_emojis]",
        "[remove_emojis, remove_punctuation]",
        "[remove_emojis, remove_tags]",
        "[remove_emojis, remove_tags, remove_punctuation]",
        "[tag_emojis]",
        "[tag_emojis, remove_punctuation]",
        "[replace_emojis]",
        "[replace_emojis_no_sep]",
        "[replace_emojis_no_sep, remove_tags]",
        "[replace_emojis_no_sep, remove_tags, remove_punctuation]",
    ]

    # BertTweetLevelExperiment using BERT (H-128) Feed
    config = {
        "max_trials": 50,
        "hyperparameters": {
            "epochs": 8,
            "batch_size": [24, 32, 48, 64, 80],
            "learning_rate": [2e-5, 3e-5, 5e-5],
            "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "Bert.hidden_size": 128,
            "Bert.preprocessing": preprocessing_choices,
            "Bert.type": "feed",
        }
    }
    experiment = BertTweetLevelExperiment(experiment_dir, "tweet_level_feed_128_1", config)
    experiment.run(x, y)

    # BertTweetLevelExperiment using BERT (H-256) Feed
    config = {
        "max_trials": 10,
        "hyperparameters": {
            "epochs": 8,
            "batch_size": [24, 32, 48, 64],
            "learning_rate": [2e-5, 3e-5, 5e-5],
            "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
            "Bert.hidden_size": 256,
            "Bert.preprocessing": "[remove_emojis, remove_tags]",
            "Bert.type": "feed",
        }
    }
    experiment = BertTweetLevelExperiment(experiment_dir, "tweet_level_feed_256_1", config)
    experiment.run(x, y)

    # BertTweetLevelExperiment using BERT (H-128) Individual
    config = {
        "max_trials": 50,
        "hyperparameters": {
            "epochs": 8,
            "batch_size": [24, 32, 48, 64, 80],
            "learning_rate": [2e-5, 3e-5, 5e-5],
            "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "Bert.hidden_size": 128,
            "Bert.preprocessing": preprocessing_choices,
            "Bert.type": "individual",
        }
    }
    experiment = BertTweetLevelExperiment(experiment_dir, "tweet_level_indiv_128_1", config)
    experiment.run(x, y)

    # BertTweetLevelFfnnExperiment using BERT (H-128) Individual
    config = {
        "max_trials": 50,
        "hyperparameters": {
            "epochs": 8,
            "batch_size": [24, 32, 48, 64, 80],
            "learning_rate": [2e-5, 3e-5, 5e-5],
            "Bert.encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "Bert.hidden_size": 128,
            "Bert.preprocessing": preprocessing_choices,
            "Bert.type": "individual",
            "Bert.trainable": True,
        }
    }
    experiment = BertTweetLevelFfnnExperiment(experiment_dir, "tweet_level_ffnn_indiv_128_1", config)
    experiment.run(x, y)

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
