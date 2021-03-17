import numpy as np
import tensorflow as tf

from base.training import allow_gpu_memory_growth
from bert_classifier.tune_bert_stats_ffnn import tune_bert_stats_ffnn
from data import load_data, BertTweetPreprocessor, parse_labels_to_floats
from bert_classifier.tune_ffnn import tune_ffnn
from bert_classifier.tune_bert_ffnn import tune_bert_ffnn
from bert_classifier.tune_bert_combined_ffnn import tune_bert_nn_classifier, tune_bert_sklearn_classifier, \
    sklearn_classifier, tune_bert_tweet_level_stats_sklearn_classifier
from statistical.tuning import get_best_trials

allow_gpu_memory_growth()


def main():

    # Load and preprocess data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data()

    tweet_train = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_train]
    tweet_val = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_val]
    tweet_test = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_test]

    label_train = parse_labels_to_floats(label_train)
    label_val = parse_labels_to_floats(label_val)
    label_test = parse_labels_to_floats(label_test)

    x_train = np.concatenate([tweet_train, tweet_val, tweet_test])
    y_train = np.concatenate([label_train, label_val, label_test])

    print("Preprocessing data")
    # tweet_preprocessor = BertTweetPreprocessor()
    # tweet_train_processed = tweet_preprocessor.transform(tweet_train)
    # tweet_val_processed = tweet_preprocessor.transform(tweet_val)
    # tweet_test_processed = tweet_preprocessor.transform(tweet_test)

    # # Tune BERT 128
    # tuner_128 = tune_bert_ffnn(
    #     x_train, y_train,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
    #     bert_size=128,
    #     project_name="bert_ffnn_17",
    #     max_trials=50,
    #     epochs=8,
    #     batch_sizes=[16, 32, 64],
    #     preprocess_data_in_training=True,
    #     model_type="individual",
    # )

    # # Tune BERT 256
    # tuner_256 = tune_bert_ffnn(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
    #     bert_size=256,
    #     project_name="bert_ffnn_13",
    #     max_trials=10,
    #     batch_sizes=[24, 32, 48, 64],
    #     tf_train_device="/cpu:0",
    # )

    # Tune BERT 128, final classifier
    # tuner_128 = tune_bert_nn_classifier(
    #     x_train, y_train,
    #     bert_size=128,
    #     project_name="bert_combined_ffnn_12",
    #     bert_model_trial_filepath="../training/bert_clf/initial_eval/bert_combined_ffnn_12/bert_model.json",
    #     max_trials=100,
    #     epochs=20,
    #     batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    # )
    #
    # tuner_128_sklearn = tune_bert_sklearn_classifier(
    #     x_train, y_train,
    #     project_name="bert_combined_ffnn_17",
    #     bert_model_trial_filepath="../training/bert_clf/initial_eval/bert_combined_ffnn_16/bert_model.json",
    #     max_trials=100,
    #     bert_model_type="individual",
    # )
    # tuner_128_stats_sklearn = tune_bert_tweet_level_stats_sklearn_classifier(
    #     x_train, y_train,
    #     project_name="bert_combined_ffnn_19",
    #     bert_model_trial_filepath="../training/bert_clf/initial_eval/bert_combined_ffnn_12/bert_model.json",
    #     max_trials=100,
    # )
    tuner_128_stats_sklearn = tune_bert_stats_ffnn(
            x_train, y_train,
            bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            bert_size=128,
            project_name="bert_ffnn_19",
            max_trials=21,
            batch_sizes=[64],
    )
    #
    # # Tune BERT 256, final classifier
    # tuner_256 = tune_bert_nn_classifier(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val, tweet_test_processed, label_test,
    #     bert_size=256,
    #     project_name="bert_combined_ffnn_13",
    #     bert_model_trial_filepath="../training/bert_clf/initial_eval/bert_ffnn_10/"
    #                               "trial_df5e725b3223e721f73bbc25a06897b4/trial.json",
    #     max_trials=200,
    #     epochs=20,
    #     batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    # )

    # print("Tuner 128:\n", tuner_128.results_summary(1))
    # print("Tuner 256:\n", tuner_256.results_summary(1))
    print("\nReadability summary:\n", get_best_trials(tuner_128_stats_sklearn).to_markdown(), sep="")


if __name__ == "__main__":
    main()


"""
TODO:
* Review BERT+FFNN results and see if added layers with Dense linear layers improve
* Look at different BERT Feed tokenization overlaps - pass data to Optimizer, which passes to fit method?
* Take the best BERT+FFNN model+hps and train a max pooling
* Finish up the statistical model
* Create a class for the statistical models
* Create evaluation classes for the separate statistical models
* Create evaluation class for the BERT model - option to choose from 2 different BERT models (128 or 256)
* Create the ensemble model and create an evaluation class for it
"""

