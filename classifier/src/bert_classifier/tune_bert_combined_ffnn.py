from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, GlobalMaxPool1D
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

    model = tf.keras.Sequential([
        Input((None, hidden_layer_size)),
        Dropout(hp.Float("dropout_rate", 0, 0.5)),
        GlobalMaxPool1D(),
        Dense(1, activation="sigmoid"),
    ])

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


def tune_bert_nn_classifier(X_train, y_train, X_val, y_val, bert_encoder_url, bert_size, project_name, bert_weights,
                            tf_train_device="/gpu:0", epochs=8, batch_sizes=None, max_trials=30):
    """ Tune a BERT final classifier """
    if batch_sizes is None:
        batch_sizes = [24, 32]

    with tf.device("/cpu:0"):
        # Create BERT Model and predict training/validation data
        print("Loading BERT model")
        bert_model, bert_tokenizer = build_base_bert(
            encoder_url=bert_encoder_url,
            trainable=False,
            hidden_layer_size=bert_size,
            tokenizer_class=BertTweetFeedTokenizer,
            return_tokenizer=True
        )
        bert_model.load_weights(bert_weights).expect_partial()

        def _bert_tokenize_predict(data):
            predictions = []
            for tweet_feed in data:
                tokenized_data = bert_tokenizer.tokenize_input(np.asarray([tweet_feed]))
                predictions.append(bert_model.predict([tokenized_data]))

            max_num_chunks = 60

            return tf.convert_to_tensor([
                tf.pad(user, paddings=[[0, max_num_chunks - len(user)], [0, 0]], constant_values=-1)
                for user in predictions
            ])

        print("Making BERT predictions")
        x_train_bert = _bert_tokenize_predict(X_train)  # shape=(num_users, 60, bert_size)
        y_train_bert = bert_tokenizer.tokenize_labels(y_train, user_label_pattern=False)
        x_val_bert = _bert_tokenize_predict(X_val)
        y_val_bert = bert_tokenizer.tokenize_labels(y_val, user_label_pattern=False)
        assert len(x_train_bert) == len(X_train)
        assert len(y_train_bert) == len(y_train)

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        hps = HyperParameters()
        hps.Choice("batch_size", batch_sizes)
        hps.Fixed("epochs", epochs)

        tuner = BayesianOptimizationTunerWithFitHyperParameters(
            hyperparameters=hps,
            hypermodel=partial(
                _build_nn_classifier,
                hidden_layer_size=bert_size,
                input_data_len=len(x_train_bert),
            ),
            objective="val_loss",
            max_trials=max_trials,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )
        print("Beginning tuning")
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
