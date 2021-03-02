from functools import partial

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from official.nlp import optimization
from kerastuner.tuners import BayesianOptimization

from bert import BaseBertWrapper, BertTweetFeedTokenizer


class FullBayesianOptimizationTuner(BayesianOptimization):
    """ BayesianOptimization eras Tuner which also tunes batch_size and epochs """

    def __init__(self, hp_batch_size, hp_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hp_batch_size = hp_batch_size
        self.hp_epochs = hp_epochs

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        fit_kwargs["batch_size"] = self.hp_batch_size(trial.hyperparameters)
        fit_kwargs["epochs"] = self.hp_epochs(trial.hyperparameters)

        super().run_trial(trial, *fit_args, **fit_kwargs)


def _build_ffnn(hp, bert_hidden_layer_size):
    model = tf.keras.models.Sequential([
        Input(shape=(bert_hidden_layer_size,), dtype=tf.float32),
        Dense(bert_hidden_layer_size * 2, activation="relu"),
        Dense(bert_hidden_layer_size, activation="relu"),
        Dense(bert_hidden_layer_size // 2, activation="relu"),
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
    bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"
    bert_hidden_layer_size = 128
    bert_weights_path = "../training/bert_feed/initial_eval/bert128-batch_size32-epochs10-lr5e-05-optimizeradamw/cp" \
                        ".ckpt"

    with tf.device("/cpu:0"):
        # Create BERT Model and predict training/validation data
        bert = BaseBertWrapper(
            encoder_url=bert_encoder_url,
            trainable=False,
            tokenizer_class=BertTweetFeedTokenizer,
            hidden_layer_size=bert_hidden_layer_size,
        )
        bert.setup_model()
        bert.load_weights(bert_weights_path)

        X_train_bert_out, y_train_bert_out = bert.predict(X_train, y_train)
        X_val_bert_out, y_val_bert_out = bert.predict(X_val, y_val)

        # Create the keras Tuner and performs a search
        tuner = FullBayesianOptimizationTuner(
            hp_batch_size=lambda hp: hp.Choice("batch_size", [16, 32, 64, 80]),
            hp_epochs=lambda hp: hp.Fixed("epochs", 50),
            hypermodel=partial(_build_ffnn, bert_hidden_layer_size=bert_hidden_layer_size),
            objective="val_loss",
            max_trials=50,
            directory="../training/bert_clf/testing/ffnn",
            project_name="bert_ffnn",
        )

        tuner.search(
            x=X_train_bert_out,
            y=y_train_bert_out,
            validation_data=(X_val_bert_out, y_val_bert_out),
            callbacks=[
                EarlyStopping("val_loss", patience=5),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs", histogram_freq=1)
            ]
        )

        return tuner
