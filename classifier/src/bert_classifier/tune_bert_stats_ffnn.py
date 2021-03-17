from functools import partial

import numpy as np
import tensorflow as tf
from kerastuner import HyperParameters
from official.nlp import optimization
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from tensorflow.keras.layers import Dense, Dropout

import data.preprocess as pre
from bert import bert_layers, BertIndividualTweetTokenizer
from bert.models import shuffle_bert_data
from experiments.tuners import BayesianOptimizationCV
from bert_classifier.tune_bert_ffnn import tokenize_data, preprocess_data, comp
from statistical.data_extraction import tweet_level_extractor

"""
Tuning objective: Train BERT by combining it with a FFNN, and evaluate the optimal batch sizes, learning
rates, and dropout rates of the model.
"""


def preprocess_and_extract_data(hp, x_train, y_train, x_test, y_test):
    # Preprocess data for tweet-level extraction
    stats_preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
    x_train_stats = stats_preprocessor.transform(x_train)
    x_test_stats = stats_preprocessor.transform(x_test)

    # Extract tweet-level statistical features
    extractor = tweet_level_extractor()
    x_train_stats = extractor.transform([tweet for tweet_feed in x_train_stats for tweet in tweet_feed])
    x_test_stats = extractor.transform([tweet for tweet_feed in x_test_stats for tweet in tweet_feed])

    # Preprocess data for BERT
    _, x_train_bert, y_train_bert, x_test_bert, y_test_bert = preprocess_data(hp, x_train, y_train, x_test, y_test)

    # Tokenize for BERT
    x_train_bert, y_train_bert, x_test_bert, y_test_bert = tokenize_data(
        hp, x_train_bert, y_train_bert, x_test_bert, y_test_bert,
        tokenizer_class=BertIndividualTweetTokenizer, shuffle=False)
    x_train_bert["tweet_level_stats"] = x_train_stats
    x_test_bert["tweet_level_stats"] = x_test_stats

    x_train_bert, y_train_bert = shuffle_bert_data(x_train_bert, y_train_bert)
    return x_train_bert, y_train_bert, x_test_bert, y_test_bert


def _build_bert_single_dense(hp):
    """ Build and compile a BERT+Dense Keras Model for hyper-parameter tuning """
    # Get BERT input and output
    bert_input, bert_output = bert_layers(
        hp.get("bert_encoder_url"),
        trainable=True,
        hidden_layer_size=hp.get("bert_size")
    )

    # bert_input["tweet_level_stats"] = tf.keras.layers.Input(
    #     (len(tweet_level_extractor().feature_names),), name="tweet_level_stats")

    # Select encoder outputs to use, and then pool them to produce 'pooled_outputs'
    # Note that encoder_outputs contains the outputs of every hidden state
    # encoder_outputs.shape == (num_encoders, batch_size, seq_len, hidden_size)
    encoder_outputs = bert_output["encoder_outputs"]
    selected_encoder_outputs = hp.Choice("selected_encoder_outputs", [
        "first_layer",
        "2nd_to_last_hidden_layer",
        "last_hidden_layer",
        "sum_all_but_last_hidden_layers",
        "sum_all_hidden_layers",
        "sum_4_2nd_to_last_hidden_layers",
        "sum_last_4_hidden_layers",
        "concat_4_2nd_to_last_hidden_layers",
        "concat_last_4_hidden_layers",
        "default",
    ])

    if selected_encoder_outputs == "default":
        dense_pooled = bert_output["pooled_output"]
    else:
        encoder_outputs = encoder_outputs[{
            "first_layer": np.s_[:1],
            "2nd_to_last_hidden_layer": np.s_[-2:-1],
            "last_hidden_layer": np.s_[-1:],
            "sum_all_but_last_hidden_layers": np.s_[:-1],
            "sum_all_hidden_layers": np.s_[:],
            "sum_4_2nd_to_last_hidden_layers": np.s_[-5:-1],
            "sum_last_4_hidden_layers": np.s_[:-4],
            "concat_4_2nd_to_last_hidden_layers": np.s_[-5:-1],
            "concat_last_4_hidden_layers": np.s_[:-4],
        }[selected_encoder_outputs]]

        # Pool encoder_outputs by summing or concatenating
        if selected_encoder_outputs.startswith("concat"):
            # Concatenate layer outputs, and extract the concatenated '[CLF]' embeddings
            # pooled_outputs.shape == (batch_size, len(encoder_outputs) * hidden_size)
            pooled_outputs = tf.concat(encoder_outputs, axis=-1)[:, 0, :]
        else:
            # Extract the '[CLF]' embeddings of each layer, and then sum them
            pooled_outputs = tf.convert_to_tensor(encoder_outputs)[:, :, 0, :]
            # pooled_outputs.shape == (batch_size, hidden_size)
            pooled_outputs = tf.reduce_sum(pooled_outputs, axis=0)

        # Pass pooled_outputs through a tanh layer
        dense_pooled = Dense(hp.get("bert_size"), activation="tanh")(pooled_outputs)

    # Classifier layer
    dropout = Dropout(hp.Fixed("dropout_rate", 0.1))(dense_pooled)
    # batch = BatchNormalization(dropout)
    dense_out = Dense(
        1, activation=hp.Fixed("dense_activation", "linear"),
        # kernel_regularizer=tf.keras.regularizers.l2(hp.Fixed("linear_kernel_reg", 0)),
        # bias_regularizer=tf.keras.regularizers.l2(hp.Fixed("linear_bias_reg", 0.01)),
        # activity_regularizer=tf.keras.regularizers.l2(hp.Fixed("linear_activity_reg", 0)),
    )(dropout)

    # Build model
    model = Model(bert_input, dense_out)
    model.compile(
        optimizer=optimization.AdamWeightDecay(learning_rate=hp.Fixed("learning_rate", 2e-5)),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )
    return model


def tune_bert_stats_ffnn(x_train, y_train, bert_encoder_url, bert_size, project_name, tf_train_device="/gpu:0",
                         epochs=8, batch_sizes=None, max_trials=30):
    """ Tune a BERT+FFNN Keras Model """
    if batch_sizes is None:
        batch_sizes = [24, 32]

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        hp = HyperParameters()
        hp.Choice("batch_size", batch_sizes)
        hp.Fixed("epochs", epochs)
        hp.Fixed("bert_encoder_url", bert_encoder_url)
        hp.Fixed("bert_size", bert_size)
        hp.Fixed("feed_data_overlap", 0)
        hp.Fixed("preprocessing", "[remove_emojis, remove_tags]")

        data_preprocessing_func = comp(partial(tokenize_data, tokenizer_class=BertIndividualTweetTokenizer),
                                       preprocess_data)  # preprocess_and_extract_data

        tuner = BayesianOptimizationCV(
            preprocess=data_preprocessing_func,
            hyperparameters=hp,
            hypermodel=_build_bert_single_dense,
            objective="val_loss",
            max_trials=max_trials,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )

        tuner.search(
            x=x_train,
            y=y_train,
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping("val_loss", patience=2),
                TensorBoard(log_dir=tuner.directory + "/" + tuner.project_name + "/logs")
            ]
        )
        return tuner
