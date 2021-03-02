import os
import time
from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from ray import tune
from ray.tune.integration.keras import TuneReportCheckpointCallback
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from bert import BaseBertWrapper, BertTweetFeedTokenizer
from models import basic_ffnn


def _trainable_ffnn(config, X_train, y_train, X_val, y_val):
    with tf.device("/cpu:0"):
        # Create BERT Model and predict training/validation data
        bert = BaseBertWrapper(
            encoder_url=config["bert_encoder_url"],
            trainable=False,
            tokenizer_class=BertTweetFeedTokenizer,
            hidden_layer_size=config["bert_hidden_layer_size"],
        )
        bert.setup_model()
        bert.load_weights(config["bert_weights_path"])

        X_train_bert_out, y_train_bert_out = bert.predict(X_train, y_train)
        X_val_bert_out, y_val_bert_out = bert.predict(X_val, y_val)

        # Create FFNN Model
        ffnn = basic_ffnn(
            bert_hidden_layer_size=config["bert_hidden_layer_size"],
            learning_rate=config["learning_rate"],
            num_training_steps=config["epochs"] * len(X_train_bert_out) // config["batch_size"]
        )

        # Train the FFNN Model
        ffnn.fit(
            x=X_train_bert_out,
            y=y_train_bert_out,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=[TuneReportCheckpointCallback(frequency=5)],
            validation_data=(X_val_bert_out, y_val_bert_out),
            use_multiprocessing=True
        )


def tune_ffnn(X_train, y_train, X_val, y_val):
    analysis = tune.run(
        partial(_trainable_ffnn, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        name="basic-ffnn--" + time.strftime("%H-%M-%S---%d-%m-%y"),
        config={
            "bert_encoder_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "bert_hidden_layer_size": 128,
            "bert_weights_path": "../training/bert_feed/initial_eval/bert128-batch_size32-epochs10-lr5e-05"
                                 "-optimizeradamw/cp.ckpt",
            "metric": "val_loss",
            "mode": "min",
            "epochs": 50,
            "learning_rate": tune.loguniform(1e-5, 0.1),
            "batch_size": tune.choice([16, 32, 64, 80]),
        },
        local_dir="training/ffnn",
        search_alg=TuneBOHB(
            max_concurrent=1,
            metric="val_loss",
            mode="min"
        ),
        scheduler=HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=50,
            reduction_factor=4,
            metric="val_loss",
            mode="min",
        ),
        keep_checkpoints_num=2,
        checkpoint_score_attr="min-val_loss",
        num_samples=50,
        checkpoint_freq=5,
    )

    print(f"""
        Tune analysis results:
        > Best result: {analysis.best_result}
        > Best trail: {analysis.best_trial}
        > Best log dir: {analysis.best_logdir}
        > Best config: \n{analysis.best_config}
        """)
    return analysis
