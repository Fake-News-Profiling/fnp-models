from functools import partial

import numpy as np
import tensorflow as tf
from kerastuner import HyperParameters, Objective
from kerastuner.oracles import BayesianOptimization as BayesianOptimizationOracle
from official.nlp import optimization
from sklearn.metrics import make_scorer, accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.layers import BatchNormalization

from bert import BertTweetFeedTokenizer, BertIndividualTweetTokenizer
from bert.models import tokenize_bert_input
from bert_classifier import BayesianOptimizationCVTunerWithFitHyperParameters
from bert_classifier.cv_tuners import SklearnCV
from bert_classifier.tune_bert_ffnn import load_bert_single_dense_model, preprocess_data
from statistical.data_extraction import combined_tweet_extractor
from statistical.tune_statistical import build_sklearn_classifier_model

"""
Tuning objective: Feed data into an already fine-tuned BERT model and extract the pooled outputs. Pool all of these 
outputs in some way, and then feed them into a FFNN for final classification.
"""


def data_preprocessing_func(hp, x_train, y_train, x_test, y_test):
    # Pool predictions for each user (resulting shape is (num_users, bert_size))
    def pool(x):
        if hp.get("pooling_type") == "max":
            return tf.convert_to_tensor([np.max(tweet_feed, axis=0) for tweet_feed in x])
        elif hp.get("pooling_type") == "min":
            return tf.convert_to_tensor([np.min(tweet_feed, axis=0) for tweet_feed in x])
        elif hp.get("pooling_type") == "average":
            return tf.convert_to_tensor([np.mean(tweet_feed, axis=0) for tweet_feed in x])
        elif hp.get("pooling_type") == "concatenate":
            return tf.convert_to_tensor([np.concatenate(tweet_feed) for tweet_feed in x])
        else:
            raise RuntimeError("Invalid 'pooling_type', should be 'average' or 'max'")

    x_train_pooled = pool(x_train)
    x_test_pooled = pool(x_test)
    return x_train_pooled, y_train, x_test_pooled, y_test


def _bert_model_wrapper(x_train, y_train, x_test, trial_filepath,
                        tf_device="/gpu:0", tokenizer_class=BertTweetFeedTokenizer, encoder_output=True):
    """
    Fits a BERT model, based on hyper-parameters from a previous trial. Note that this uses the
    `load_bert_single_dense_model` function from `tune_bert_ffnn.py`.
    Once trained, the BERT layers are extracted and used to predict `x_train` and `x_test` (which are returned).
    """
    with tf.device(tf_device):
        bert_model, hp = load_bert_single_dense_model(trial_filepath)

        # Preprocess the data
        _, x_train, y_train, x_test, y_test = preprocess_data(hp, x_train, y_train, x_test, None)

        # Tokenize training data and train this BERT model
        x_train_bert, y_train_bert, tokenizer = tokenize_bert_input(
            hp.get("bert_encoder_url"), hp.get("bert_size"), tokenizer_class, x_train, y_train,
            shuffle=True, feed_overlap=hp.get("feed_data_overlap"), return_tokenizer=True)

        bert_model.fit(
            x=x_train_bert,
            y=y_train_bert,
            epochs=hp.get("epochs"),
            batch_size=hp.get("batch_size"),
        )
        if encoder_output:
            predict_model = Model(bert_model.inputs, bert_model.layers[3].output["pooled_output"])
        else:
            predict_model = bert_model

        # Transform the input data and pass it through BERT
        def _bert_tokenize_predict(data):
            return np.asarray([
                predict_model.predict(
                    tokenizer.tokenize_input([tweet_feed], overlap=hp.get("feed_data_overlap"))
                ) for tweet_feed in data
            ])

        x_train_out = _bert_tokenize_predict(x_train)
        x_test_out = _bert_tokenize_predict(x_test)
        assert len(x_train_out) == len(x_train)
        assert len(x_test_out) == len(x_test)
        return x_train_out, x_test_out


def _bert_tweet_level_stats_wrapper(x_train, y_train, x_test, *args, **kwargs):
    """
    Fits a BERT model and extracts BERT (Individual) tweet-level embeddings, as well as statistical tweet-level
    embeddings, returning the concatenated embeddings for each tweet
    """
    kwargs["tokenizer_class"] = BertIndividualTweetTokenizer
    tl_extractor = tweet_level_extractor()

    def tl_extractor_wrapper(data):
        return np.asarray([tl_extractor.transform(tweet_feed) for tweet_feed in data])

    x_train_stats, x_test_stats = tl_extractor_wrapper(x_train), tl_extractor_wrapper(x_test)
    x_train_bert, x_test_bert = _bert_model_wrapper(x_train, y_train, x_test, *args, **kwargs)
    # shapes = (num_users, num_tweets, embedding_size)
    # Transform to (num_users, num_tweets, bert_embedding_size + stats_embedding_size)
    x_train_out = np.concatenate([x_train_stats, x_train_bert], axis=-1)
    x_test_out = np.concatenate([x_test_stats, x_test_bert], axis=-1)
    return x_train_out, x_test_out


def build_nn_classifier(hp):
    """
    Build a neural network classifier that takes in BERT pooled_outputs, pools them, and passes them to a
    classification layer for user-level classification
    """
    # Model architecture
    inputs = Input((hp.get("bert_size"),))
    dropout_last = Dropout(hp.Float("last_dropout_rate", 0, 0.5))(inputs)
    batch_last = BatchNormalization()(dropout_last)
    linear_last = Dense(
        1, activation=hp.Fixed("last_dense_activation", "linear"),
        kernel_regularizer=tf.keras.regularizers.l2(hp.Choice("linear_kernel_reg", [0.0001, 0.001, 0.01, 0.02])),
        bias_regularizer=tf.keras.regularizers.l2(hp.Choice("linear_bias_reg", [0.0001, 0.001, 0.01, 0.02])),
        activity_regularizer=tf.keras.regularizers.l2(hp.Choice("linear_activity_reg", [0.0001, 0.001, 0.01, 0.02])),
    )(batch_last)

    # Build model
    model = Model(inputs, linear_last)
    model.compile(
        optimizer=optimization.AdamWeightDecay(
            learning_rate=hp.Float("learning_rate", 1e-6, 1e-3, sampling="log"),
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )
    return model


def tune_bert_nn_classifier(x_train, y_train, bert_size, project_name, bert_model_trial_filepath,
                            tf_train_device="/gpu:0", epochs=8, batch_sizes=None, max_trials=30,
                            bert_model_type="feed"):
    """ Tune a BERT final classifier """
    if batch_sizes is None:
        batch_sizes = [24, 32]

    # Create the keras Tuner and performs a search
    with tf.device(tf_train_device):
        print("Building tuner")
        hp = HyperParameters()
        hp.Choice("batch_size", batch_sizes)
        hp.Fixed("epochs", epochs)
        hp.Choice("pooling_type", ["max", "min", "average"])
        hp.Fixed("bert_size", bert_size)

        tuner = BayesianOptimizationCVTunerWithFitHyperParameters(
            preprocess=data_preprocessing_func,
            hyperparameters=hp,
            hypermodel=build_nn_classifier,
            objective="val_loss",
            max_trials=max_trials,
            directory="../training/bert_clf/initial_eval",
            project_name=project_name,
        )

        print("Fitting tuner with training data")
        tuner.fit_data(
            x_train, y_train,
            partial(_bert_model_wrapper,
                    trial_filepath=bert_model_trial_filepath,
                    tokenizer_class=BertTweetFeedTokenizer
                    if bert_model_type == "feed" else BertIndividualTweetTokenizer)
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


def tune_bert_sklearn_classifier(x_train, y_train, project_name, bert_model_trial_filepath, max_trials=30,
                                 bert_model_type="feed"):
    """ Tune a BERT final classifier """
    # Generate user-level data from BERT
    print("Building tuner")
    tuner = sklearn_classifier(project_name, max_trials=max_trials)

    print("Fitting tuner with training data")
    tuner.fit_data(
        x_train, y_train,
        partial(_bert_model_wrapper,
                trial_filepath=bert_model_trial_filepath,
                tokenizer_class=BertTweetFeedTokenizer if bert_model_type == "feed" else BertIndividualTweetTokenizer,
                encoder_output=True,
                )
    )

    print("Beginning tuning")
    tuner.search(x_train, y_train)
    return tuner


def tune_bert_tweet_level_stats_sklearn_classifier(x_train, y_train, project_name, bert_model_trial_filepath,
                                                   max_trials=30):
    """
    Tune a BERT final classifier, where BERT (Individual) outputs are combined with tweet-level statistical
    information
    """
    print("Building tuner")
    tuner = sklearn_classifier(project_name, max_trials=max_trials, pooling_types=["concatenate", "mean", "max"])

    print("Fitting tuner with training data")
    tuner.fit_data(
        x_train, y_train,
        partial(_bert_tweet_level_stats_wrapper,
                trial_filepath=bert_model_trial_filepath,
                encoder_output=True)
    )

    print("Beginning tuning")
    tuner.search(x_train, y_train)
    return tuner


def sklearn_classifier(project_name, max_trials=30, pooling_types=None):
    hp = HyperParameters()
    if pooling_types is None:
        pooling_types = ["concatenate"]
    hp.Choice("pooling_type", pooling_types)

    return SklearnCV(
        preprocess=data_preprocessing_func,
        oracle=BayesianOptimizationOracle(
            objective=Objective("score", "min"),  # minimise log loss
            max_trials=max_trials,
            hyperparameters=hp,
        ),
        hypermodel=build_sklearn_classifier_model,
        scoring=make_scorer(log_loss, needs_proba=True),
        metrics=[accuracy_score, f1_score],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3),
        directory="../training/bert_clf/initial_eval",
        project_name=project_name,
    )
