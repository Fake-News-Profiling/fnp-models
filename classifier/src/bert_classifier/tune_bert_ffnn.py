from functools import partial

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from official.nlp import optimization
from kerastuner import HyperParameters

from bert import BertTweetFeedTokenizer, bert_layers
from bert_classifier import BayesianOptimizationTunerWithFitHyperParameters


def _build_bert_ffnn_varying_layers(hp, hidden_layer_size, bert_input, bert_output, input_data_len):
    """ Build and compile a BERT+FFNN Keras Model for hyper-parameter tuning """

    # FFNN with num of layers and num neurons as hyper-parameters
    # BERT -> Drop -> ((BatchNorm)? -> Dense -> Drop)? -> (BatchNorm)? -> Dense
    num_layers = hp.Int("num_layers", 1, 4)
    include_batch_norm = hp.Boolean("include_batch_norm")
    dense_1_out = hp.Int("dense_1_out", hidden_layer_size // 4, hidden_layer_size * 2, step=hidden_layer_size // 4)

    def ff_layer(_prev_layer, dense_units, i):
        if include_batch_norm:
            _prev_layer = BatchNormalization()(_prev_layer)

        dense = Dense(dense_units, activation="relu")(_prev_layer)
        drop = Dropout(hp.Float(f"drop_rate_{i}", 0.01, 0.6, sampling="linear"))(dense)
        return drop

    prev_layer = Dropout(hp.Float(f"drop_rate_0", 0.01, 0.5, sampling="linear"))(bert_output["pooled_output"])
    for i in range(1, num_layers):
        prev_layer = ff_layer(prev_layer, dense_1_out // max(1, i * 2), i)

    if include_batch_norm:
        prev_layer = BatchNormalization()(prev_layer)

    last_layer = Dense(1, activation="sigmoid")(prev_layer)

    # Build model
    model = Model(bert_input, last_layer)
    num_train_steps = hp.get("epochs") * input_data_len // hp.get("batch_size")
    optimizer = optimization.create_optimizer(
        init_lr=hp.Float("learning_rate", 2e-5, 1e-4, sampling="log"),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_train_steps // hp.get("epochs"),
        optimizer_type='adamw',
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )
    return model


def _build_bert_ffnn_3_layers(hp, hidden_layer_size, bert_input, bert_output, input_data_len):
    """ Build and compile a BERT+FFNN (3 layers) Keras Model for hyper-parameter tuning """

    # FFNN with num of layers and num neurons as hyper-parameters
    # BERT -> (Drop -> BatchNorm -> Dense){3}
    def ff_layer(_prev_layer, dense_units, _i, activation="relu"):
        if hp.Boolean("dropout"):
            _prev_layer = Dropout(hp.Float(f"drop_rate_{_i}", 0.3, 0.5, sampling="log"))(_prev_layer)
        if hp.Boolean("batch_norm"):
            _prev_layer = BatchNormalization()(_prev_layer)
        dense = Dense(dense_units, activation=activation)(_prev_layer)
        return dense

    prev_layer = bert_output["pooled_output"]
    # dense_0_unit = hp.Choice("dense_0_unit", [hidden_layer_size * 2, hidden_layer_size, hidden_layer_size // 2])
    dense_0_unit = hp.Choice("dense_0_unit", [hidden_layer_size, hidden_layer_size // 2])

    for i in range(1):
        prev_layer = ff_layer(prev_layer, dense_0_unit // max(1, i * 2), i)

    last_layer = ff_layer(prev_layer, 1, 2, activation="sigmoid")

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


def tune_bert_ffnn(X_train, y_train, X_val, y_val):
    """ Tune a BERT+FFNN Keras Model """
    bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"
    bert_hidden_layer_size = 128

    with tf.device("/cpu:0"):
        # Create BERT Model and predict training/validation data
        bert_input, bert_output, x_train_bert, y_train_bert, x_val_bert, y_val_bert = bert_layers(
            encoder_url=bert_encoder_url,
            trainable=True,
            hidden_layer_size=bert_hidden_layer_size,
            tokenizer_class=BertTweetFeedTokenizer,
            data_train=(X_train, y_train),
            data_val=(X_val, y_val),
            shuffle_data=True,
        )

    def _tune_search(batch_sizes, project_name):
        # Create the keras Tuner and performs a search
        hps = HyperParameters()
        hps.Choice("batch_size", batch_sizes)
        hps.Fixed("epochs", 16)

        tuner = BayesianOptimizationTunerWithFitHyperParameters(
            hyperparameters=hps,
            hypermodel=partial(
                _build_bert_ffnn_3_layers,
                hidden_layer_size=bert_hidden_layer_size,
                bert_input=bert_input,
                bert_output=bert_output,
                input_data_len=len(x_train_bert["input_word_ids"]),
            ),
            objective="val_loss",
            max_trials=25,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )
        tuner.search(
            x=x_train_bert,
            y=y_train_bert,
            validation_data=(x_val_bert, y_val_bert),
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=3),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs")
            ]
        )
        return tuner

    # Tune search for batch sizes in [16, 32] (on GPU)
    with tf.device("/gpu:0"):
        _tune_search([32], "bert_ffnn_4")

    # Tune search for batch sizes in [64, 80] (on CPU)
    # with tf.device("/cpu:0"):
    #     _tune_search([64, 48], "bert_ffnn_2")
