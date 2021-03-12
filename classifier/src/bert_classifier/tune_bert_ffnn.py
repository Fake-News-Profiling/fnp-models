from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from official.nlp import optimization
from kerastuner import HyperParameters

from base import load_hyperparameters
from bert import BertTweetFeedTokenizer, bert_layers
from bert.models import tokenize_bert_input
from bert_classifier import BayesianOptimizationTunerWithFitHyperParameters


"""
Tuning objective: Train BERT by combining it with a FFNN, and evaluate the optimal batch sizes, learning
rates, and dropout rates of the model.
"""


def data_preprocessing_func(hps, x_train, y_train, x_test, y_test):
    return tokenize_bert_input(
        encoder_url=hps.get("bert_encoder_url"),
        hidden_layer_size=hps.get("bert_size"),
        tokenizer_class=BertTweetFeedTokenizer,
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        shuffle=True,
        feed_overlap=hps.get("feed_data_overlap"),
    )


def _build_bert_single_dense(hp):
    """ Build and compile a BERT+Dense Keras Model for hyper-parameter tuning """
    # Get BERT input and output
    bert_input, bert_output = bert_layers(
        hp.get("bert_encoder_url"),
        trainable=True,
        hidden_layer_size=hp.get("bert_size")
    )
    dropout_1 = Dropout(hp.Float("dropout_rate_1"), 0, 0.5)(bert_output["pooled_output"])
    batch_1 = BatchNormalization(dropout_1)
    relu_1 = Dense(
        hp.get("bert_size"), activation=hp.Fixed("dense_1_activation", "relu"),
        kernel_regularizer=tf.keras.regularizers.l2(hp.Choice("dense_1_kernel_reg", [0.0001, 0.001, 0.01, 0.02])),
        bias_regularizer=tf.keras.regularizers.l2(hp.Choice("dense_1_bias_reg", [0.0001, 0.001, 0.01, 0.02])),
        activity_regularizer=tf.keras.regularizers.l2(hp.Choice("dense_1_activity_reg", [0.0001, 0.001, 0.01, 0.02])),
    )(batch_1)
    dropout_2 = Dropout(hp.Float("dropout_rate", 0, 0.5))(relu_1)
    batch_2 = BatchNormalization()(dropout_2)
    linear_2 = Dense(
        1, activation=hp.Fixed("dense_activation", "linear"),
        kernel_regularizer=tf.keras.regularizers.l2(hp.Choice("linear_kernel_reg", [0.0001, 0.001, 0.01, 0.02])),
        bias_regularizer=tf.keras.regularizers.l2(hp.Choice("linear_bias_reg", [0.0001, 0.001, 0.01, 0.02])),
        activity_regularizer=tf.keras.regularizers.l2(hp.Choice("linear_activity_reg", [0.0001, 0.001, 0.01, 0.02])),
    )(batch_2)

    # Build model
    model = Model(bert_input, linear_2)
    num_train_steps = hp.get("epochs") * hp.get("input_data_len") // hp.get("batch_size")
    optimizer = optimization.create_optimizer(
        init_lr=hp.Choice("learning_rate", [2e-5, 3e-5, 5e-5]),
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


def load_bert_single_dense_model(trial_filepath):
    hps = load_hyperparameters(trial_filepath)
    return _build_bert_single_dense(hps), hps


def _build_bert_ffnn(hp, hidden_layer_size, bert_input, bert_output, input_data_len):
    """ Build and compile a BERT+FFNN (3 layers) Keras Model for hyper-parameter tuning """

    # FFNN with num of layers and num neurons as hyper-parameters
    # BERT -> (Drop -> BatchNorm -> Dense){num_layers}
    dropout_rate = hp.Float("dropout_rate", 0.2, 0.3, sampling="log")
    print(bert_output)

    def ff_layer(_prev_layer, dense_units, activation="relu"):
        drop = Dropout(dropout_rate)(_prev_layer)
        batch = BatchNormalization()(drop)
        dense = Dense(dense_units, activation=activation)(batch)
        return dense

    prev_layer = bert_output["pooled_output"]
    dense_0_unit = hp.Choice("dense_0_unit", [hidden_layer_size // 2, 3 * hidden_layer_size // 4, hidden_layer_size])

    for i in range(hp.Int("num_layers", 0, 3)):
        prev_layer = ff_layer(prev_layer, dense_0_unit // max(1, i * 2))

    last_layer = ff_layer(prev_layer, 1, activation="sigmoid")

    # Build model
    model = Model(bert_input, last_layer)
    num_train_steps = hp.get("epochs") * input_data_len // hp.get("batch_size")
    optimizer = optimization.create_optimizer(
        init_lr=hp.Fixed("learning_rate", 5e-5),
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


def tune_bert_ffnn(x_train, y_train, x_val, y_val, bert_encoder_url, bert_size, project_name, tf_train_device="/gpu:0",
                   epochs=8, batch_sizes=None, max_trials=30):
    """ Tune a BERT+FFNN Keras Model """
    if batch_sizes is None:
        batch_sizes = [24, 32]

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        hps = HyperParameters()
        hps.Choice("batch_size", batch_sizes)
        hps.Fixed("epochs", epochs)
        hps.Fixed("bert_encoder_url", bert_encoder_url)
        hps.Fixed("bert_size", bert_size)
        hps.Choice("feed_data_overlap", [0, 50])

        tuner = BayesianOptimizationTunerWithFitHyperParameters(
            data_preprocessing_func=data_preprocessing_func,
            hyperparameters=hps,
            hypermodel=_build_bert_single_dense,
            objective="val_loss",
            max_trials=max_trials,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )

        x_train_for_cv = np.concatenate([x_train, x_val])
        y_train_for_cv = np.concatenate([y_train, y_val])
        tuner.search(
            x=x_train_for_cv,
            y=y_train_for_cv,
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=2),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs")
            ]
        )
        return tuner


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