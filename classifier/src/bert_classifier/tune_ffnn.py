from functools import partial

import tensorflow as tf
from kerastuner import HyperParameters
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from bert import BertTweetFeedTokenizer, build_base_bert
from experiments.tuners import BayesianOptimizationCV

"""
Tuning objective: Feed predictions from an already fine-tuned BERT model into a FFNN, and train the FFNN on the 
classification task.
"""


def _build_ffnn(hp, bert_hidden_layer_size):
    """ Build and compile a FFNN Keras Model for hyper-parameter tuning """
    model = tf.keras.models.Sequential([
        # Input
        Input(shape=(bert_hidden_layer_size,), dtype=tf.float32),

        # Layer 1
        BatchNormalization(),
        Dense(bert_hidden_layer_size * 2, activation="relu"),
        Dropout(hp.Float("rate", 0.01, 0.5, sampling="linear")),

        # Layer 2
        BatchNormalization(),
        Dense(bert_hidden_layer_size, activation="relu"),
        Dropout(hp.Float("rate", 0.01, 0.5, sampling="linear")),

        # Layer 3
        BatchNormalization(),
        Dense(bert_hidden_layer_size // 2, activation="relu"),
        Dropout(hp.Float("rate", 0.01, 0.5, sampling="linear")),

        # Output layer
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ])

    optimizer = tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-5, 1e-2, sampling="log"))
    # optimizer = optimization.create_optimizer(
    #     learning_rate, num_training_steps, int(warmup_rate * num_training_steps), optimizer_type='adamw')
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )

    return model


def tune_ffnn(X_train, y_train, X_val, y_val):
    """ Tune a FFNN Keras Model for BERT outputs """
    bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"
    bert_hidden_layer_size = 128
    bert_weights_path = "../training/bert_feed/initial_eval/bert128-batch_size32-epochs10-lr5e-05-optimizeradamw/cp" \
                        ".ckpt"

    with tf.device("/cpu:0"):
        # Create BERT Model and predict training/validation data
        bert, x_train_bert, y_train_bert, x_val_bert, y_val_bert = build_base_bert(
            encoder_url=bert_encoder_url,
            trainable=False,
            hidden_layer_size=bert_hidden_layer_size,
            tokenizer_class=BertTweetFeedTokenizer,
            data_train=(X_train, y_train),
            data_val=(X_val, y_val),
        )
        bert.load_weights(bert_weights_path).expect_partial()
        hps = HyperParameters()
        hps.Choice("batch_size", [16, 32, 64, 80])
        hps.Fixed("epochs", 50)

        tuner = BayesianOptimizationCV(
            hyperparameters=hps,
            hypermodel=partial(_build_ffnn, bert_hidden_layer_size=bert_hidden_layer_size),
            objective="val_loss",
            max_trials=50,
            directory="../training/bert_clf/initial_eval",
            project_name="ffnn",
        )

        tuner.search(
            x=x_train_bert,
            y=y_train_bert,
            validation_data=(x_val_bert, y_val_bert),
            callbacks=[
                EarlyStopping("val_loss", patience=2),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs", histogram_freq=1)
            ]
        )

        return tuner
