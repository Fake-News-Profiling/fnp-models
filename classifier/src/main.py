import tensorflow as tf

from base.training import allow_gpu_memory_growth
from data import load_data, BertTweetPreprocessor, parse_labels_to_floats
from bert_classifier.tune_ffnn import tune_ffnn
from bert_classifier.tune_bert_ffnn import tune_bert_ffnn
from bert_classifier.tune_bert_combined_ffnn import tune_bert_nn_classifier

allow_gpu_memory_growth()


def main():

    # Load and preprocess data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data()

    tweet_train = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_train]
    tweet_val = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_val]
    tweet_test = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_test]

    print("Preprocessing data")
    tweet_preprocessor = BertTweetPreprocessor()
    tweet_train_processed = tweet_preprocessor.transform(tweet_train)
    tweet_val_processed = tweet_preprocessor.transform(tweet_val)
    tweet_test_processed = tweet_preprocessor.transform(tweet_test)
    label_train = parse_labels_to_floats(label_train)
    label_val = parse_labels_to_floats(label_val)
    label_test = parse_labels_to_floats(label_test)

    # Tune BERT 128
    tuner_128 = tune_bert_ffnn(
        tweet_train_processed, label_train, tweet_val_processed, label_val,
        bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        bert_size=128,
        project_name="bert_ffnn_14",
        max_trials=30,
        epochs=8,
        batch_sizes=[8, 16, 32, 64],
    )

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

    # # Tune BERT 128, final classifier
    # tuner_128 = tune_bert_nn_classifier(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val, tweet_test_processed, label_test,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
    #     bert_size=128,
    #     project_name="bert_combined_ffnn_12",
    #     bert_model_trial_filepath="../training/bert_clf/initial_eval/bert_ffnn_6/trial_0b51ec0b3a25404c4fb4ee3bdeaa8d66"
    #                               "/trial.json",
    #     max_trials=200,
    #     epochs=20,
    #     batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    # )
    #
    # # Tune BERT 256, final classifier
    # tuner_256 = tune_bert_nn_classifier(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val, tweet_test_processed, label_test,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
    #     bert_size=256,
    #     project_name="bert_combined_ffnn_13",
    #     bert_model_trial_filepath="../training/bert_clf/initial_eval/bert_ffnn_10/"
    #                               "trial_df5e725b3223e721f73bbc25a06897b4/trial.json",
    #     max_trials=200,
    #     epochs=20,
    #     batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    # )

    print("Tuner 128:\n", tuner_128.results_summary(1))
    # print("Tuner 256:\n", tuner_256.results_summary(1))


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

