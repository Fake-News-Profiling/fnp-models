from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from official.nlp import optimization
from kerastuner import HyperParameters
from tensorflow_hub import KerasLayer

from bert import BertTweetFeedTokenizer, build_base_bert
from bert.models import tokenize_bert_input
from bert_classifier import BayesianOptimizationTunerWithFitHyperParameters
from bert_classifier.tune_bert_ffnn import load_bert_single_dense_model

"""
Tuning objective: Feed data into an already fine-tuned BERT model and extract the pooled outputs. Pool all of these 
outputs in some way, and then feed them into a FFNN for final classification.
"""


class BayesianOptimizationTunerWithBertPredictionData(BayesianOptimizationTunerWithFitHyperParameters):
    """ BayesianOptimization keras Tuner which fits the tuner with some data to pass through a trained BERT model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train = self.y_train = None

    def fit_data(self, x_train, y_train, bert_model_wrapper):
        """
        Splits the data into n_splits folds, trains BERT on each training fold, and saves the data for cross-validation
        """
        hps = self.oracle.get_space()

        # Split data and train BERTs
        data_splits = []
        for x_train, y_train, x_test, y_test in self.kfold_split_wrapper(x_train, y_train):
            x_train_bert, x_test_bert = bert_model_wrapper(x_train, y_train, x_test)
            data_splits.append((x_train_bert, y_train, x_test_bert, y_test))

        self.fit_cv_data(data_splits)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # Pool predictions for each user (resulting shape is (num_users, bert_size))
        if trial.hyperparameters.get("pooling_type") == "max":
            x = tf.convert_to_tensor([np.max(tweet_feed, axis=0) for tweet_feed in self.x_train])
        elif trial.hyperparameters.get("pooling_type") == "average":
            x = tf.convert_to_tensor([np.mean(tweet_feed, axis=0) for tweet_feed in self.x_train])
        else:
            raise RuntimeError("Invalid 'pooling_type', should be 'average' or 'max'")

        fit_kwargs["x"] = x
        fit_kwargs["y"] = self.y_train
        super().run_trial(trial, *fit_args, **fit_kwargs)


def _bert_model_wrapper(x_train, y_train, x_test, trial_filepath):
    with tf.device("/cpu:0"):
        bert_model, hps = load_bert_single_dense_model(
            trial_filepath=trial_filepath,
            input_data_len=[[len(tweet) for tweet in tweet_feed] for tweet_feed in x_train],
        )

        # Tokenize training data and train this BERT model
        x_train_bert, y_train_bert, x_test_bert, _, tokenizer = tokenize_bert_input(
            hps.get("bert_encoder_url"), hps.get("bert_size"), BertTweetFeedTokenizer, x_train, y_train,
            x_test, None, shuffle=True, feed_overlap=hps.get("feed_data_overlap"), return_tokenizer=True)

        bert_model.fit(
            x=x_train_bert,
            y=y_train_bert,
            epochs=hps.get("epochs"),
            batch_size=hps.get("batch_size"),
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=1),
            ],
        )

        # Transform the input data and pass it through BERT
        def _bert_tokenize_predict(data):
            # Make predictions
            predictions = []
            for tweet_feed in data:
                tokenized_data = tokenizer.tokenize_input(
                    np.asarray([tweet_feed]), overlap=hps.get("feed_data_overlap"))
                predictions.append(bert_model.predict([tokenized_data]))

            return predictions

        x_train_out = _bert_tokenize_predict(x_train)
        x_test_out = _bert_tokenize_predict(x_test)
        assert len(x_train_out) == len(x_train)
        assert len(x_test_out) == len(x_test)
        return x_train_out, x_test_out


def _build_nn_classifier(hp, input_data_len):
    """
    Build a neural network classifier that takes in BERT pooled_outputs, pools them, and passes them to a
    classification layer for user-level classification
    """

    # Get BERT input and output
    inputs = Input((hp.get("bert_size"),))
    dropout = Dropout(hp.Float("dropout_rate", 0, 0.5))(inputs)

    kernel_reg = None if hp.Boolean("use_dense_kernel_reg") else hp.Choice("dense_kernel_reg", ["l1", "l2"])
    bias_reg = None if hp.Boolean("use_dense_bias_reg") else hp.Choice("dense_bias_reg", ["l1", "l2"])
    dense_clf = Dense(
        units=1,
        activation=hp.Choice("dense_activation", ["linear", "sigmoid"]),
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
    )(dropout)

    model = Model(inputs, dense_clf)

    num_train_steps = hp.get("epochs") * input_data_len // hp.get("batch_size")
    optimizer = optimization.create_optimizer(
        init_lr=hp.Float("learning_rate", 1e-6, 1e-3, sampling="log"),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_train_steps // 10,
        optimizer_type='adamw',
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )
    return model


def tune_bert_nn_classifier(x_train, y_train, x_val, y_val, x_test, y_test, bert_encoder_url, bert_size, project_name,
                            bert_model_trial_filepath, tf_train_device="/gpu:0", epochs=8, batch_sizes=None,
                            max_trials=30):
    """ Tune a BERT final classifier """
    if batch_sizes is None:
        batch_sizes = [24, 32]

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        print("Building tuner")
        hps = HyperParameters()
        hps.Choice("batch_size", batch_sizes)
        hps.Fixed("epochs", epochs)
        hps.Fixed("bert_encoder_url", bert_encoder_url)
        hps.Fixed("bert_size", bert_size)
        hps.Fixed("feed_data_overlap", 50)
        hps.Choice("pooling_type", ["max", "average"])

        tuner = BayesianOptimizationTunerWithBertPredictionData(
            hyperparameters=hps,
            hypermodel=partial(
                _build_nn_classifier,
                input_data_len=len(x_train),
            ),
            objective="val_loss",
            max_trials=max_trials,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )

        print("Fitting tuner with training data")
        x_train_for_cv = np.concatenate([x_train, x_val])
        y_train_for_cv = np.concatenate([y_train, y_val])
        tuner.fit_data(
            x_train_for_cv, y_train_for_cv,
            partial(_bert_model_wrapper, trial_filepath=bert_model_trial_filepath)
        )

        print("Beginning tuning")
        tuner.search(
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=2),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs")
            ]
        )

        return tuner
