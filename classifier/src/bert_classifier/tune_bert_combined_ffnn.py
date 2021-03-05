from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from official.nlp import optimization
from kerastuner import HyperParameters

from bert import BertTweetFeedTokenizer, build_base_bert
from bert_classifier import BayesianOptimizationTunerWithFitHyperParameters


"""
Tuning objective: Feed data into an already fine-tuned BERT model and extract the pooled outputs. Pool all of these 
outputs in some way, and then feed them into a FFNN for final classification.
"""


def _build_nn_classifier(hp, hidden_layer_size, input_data_len):
    """
    Build a neural network classifier that takes in BERT pooled_outputs, pools them, and passes them to a
    classification layer for user-level classification
    """

    # FFNN with num of layers and num neurons as hyper-parameters
    # BERT -> (Drop -> BatchNorm -> Dense){2}
    dropout_rate = hp.Float(f"dropout_rate", 0.2, 0.3, sampling="log")

    def ff_layer(_prev_layer, dense_units, activation="relu"):
        drop = Dropout(dropout_rate)(_prev_layer)
        batch = BatchNormalization()(drop)
        dense = Dense(dense_units, activation=activation)(batch)
        return dense

    prev_layer = bert_output["pooled_output"]
    dense_0_unit = hp.Choice("dense_0_unit", [hidden_layer_size // 2, 3 * hidden_layer_size // 4, hidden_layer_size])

    for i in range(2):
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


def tune_bert_ffnn(X_train, y_train, X_val, y_val, bert_encoder_url, bert_size, project_name, tf_train_device="/gpu:0",
                   epochs=8, batch_sizes=None, max_trials=30):
    """ Tune a BERT+FFNN Keras Model """
    if batch_sizes is None:
        batch_sizes = [24, 32]

    with tf.device("/cpu:0"):
        # Create BERT Model and predict training/validation data
        bert_model, bert_tokenizer = build_base_bert(
            encoder_url=bert_encoder_url,
            trainable=False,
            hidden_layer_size=bert_size,
            tokenizer_class=BertTweetFeedTokenizer,
            return_tokenizer=True
        )

        def _bert_tokenize_predict(data):
            predictions = []
            for tweet_feed in data:
                tokenized_data = bert_tokenizer.tokenize_input(np.asarray([tweet_feed]))
                predictions.append(bert_model.predict([tokenized_data]))

            return predictions

        x_train_bert = _bert_tokenize_predict(X_train)  # shape=(num_users, num_tweets, bert_size)
        x_val_bert = _bert_tokenize_predict(X_val)

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        hps = HyperParameters()
        hps.Choice("batch_size", batch_sizes)
        hps.Fixed("epochs", epochs)

        tuner = BayesianOptimizationTunerWithFitHyperParameters(
            hyperparameters=hps,
            hypermodel=partial(
                _build_ffnn,
                hidden_layer_size=bert_size,
                bert_input=bert_input,
                bert_output=bert_output,
                input_data_len=len(x_train_bert["input_word_ids"]),
            ),
            objective="val_loss",
            max_trials=max_trials,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )
        tuner.search(
            x=x_train_bert,
            y=y_train_bert,
            validation_data=(x_val_bert, y_val_bert),
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=2),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs")
            ]
        )
        return tuner
