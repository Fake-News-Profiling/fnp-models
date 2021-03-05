from functools import partial

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from official.nlp import optimization
from kerastuner import HyperParameters

from bert import BertTweetFeedTokenizer, bert_layers
from bert_classifier import BayesianOptimizationTunerWithFitHyperParameters


"""
Tuning objective: Train BERT by combining it with a FFNN, and evaluate the optimal batch sizes, learning
rates, and dropout rates of the model.
"""


def _build_bert_ffnn(hp, hidden_layer_size, bert_input, bert_output, input_data_len):
    """ Build and compile a BERT+FFNN (3 layers) Keras Model for hyper-parameter tuning """

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
        # Create BERT Model
        bert_input, bert_output, x_train_bert, y_train_bert, x_val_bert, y_val_bert = bert_layers(
            encoder_url=bert_encoder_url,
            trainable=True,
            hidden_layer_size=bert_size,
            tokenizer_class=BertTweetFeedTokenizer,
            data_train=(X_train, y_train),
            data_val=(X_val, y_val),
            shuffle_data=True,
        )

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        hps = HyperParameters()
        hps.Choice("batch_size", batch_sizes)
        hps.Fixed("epochs", epochs)

        tuner = BayesianOptimizationTunerWithFitHyperParameters(
            hyperparameters=hps,
            hypermodel=partial(
                _build_bert_ffnn,
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
"""