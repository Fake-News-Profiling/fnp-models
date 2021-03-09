import tensorflow as tf

from data import load_data, BertTweetFeedDataPreprocessor
from bert_classifier.tune_ffnn import tune_ffnn
from bert_classifier.tune_bert_ffnn import tune_bert_ffnn
from bert_classifier.tune_bert_combined_ffnn import tune_bert_nn_classifier

# Allow GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def main():

    # Load and preprocess data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data()

    print("Preprocessing data")
    tweet_preprocessor = BertTweetFeedDataPreprocessor()
    tweet_train_processed = tweet_preprocessor.transform(tweet_train)
    tweet_val_processed = tweet_preprocessor.transform(tweet_val)
    tweet_test_processed = tweet_preprocessor.transform(tweet_test)

    # # Tune BERT 128
    # tuner_128 = tune_bert_ffnn(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
    #     bert_size=128,
    #     project_name="bert_ffnn_11",
    #     batch_sizes=[24, 32, 48, 64]
    # )
    #
    # # Tune BERT 256
    # tuner_256 = tune_bert_ffnn(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
    #     bert_size=256,
    #     project_name="bert_ffnn_12",
    #     tf_train_device="/cpu:0",
    #     max_trials=10,
    #     batch_sizes=[24, 32, 48, 64]
    # )

    # # Tune BERT 128, final classifier
    # tuner_128 = tune_bert_nn_classifier(
    #     tweet_train_processed, label_train, tweet_val_processed, label_val, tweet_test_processed, label_test,
    #     bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
    #     bert_size=128,
    #     project_name="bert_combined_ffnn_12",
    #     bert_weights="../training/bert_clf/initial_eval/bert_ffnn_6/trial_0b51ec0b3a25404c4fb4ee3bdeaa8d66/checkpoints/"
    #                  "epoch_0/checkpoint",
    #     max_trials=200,
    #     epochs=20,
    #     batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    # )

    # Tune BERT 256, final classifier
    tuner_256 = tune_bert_nn_classifier(
        tweet_train_processed, label_train, tweet_val_processed, label_val, tweet_test_processed, label_test,
        bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
        bert_size=256,
        project_name="bert_combined_ffnn_13",
        bert_weights="../training/bert_clf/initial_eval/bert_ffnn_9/trial_2481cabc193f7083ddce8cf11025bc8d/checkpoints/"
                     "epoch_0/checkpoint",
        max_trials=200,
        epochs=20,
        batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    )

    # print("Tuner 128:\n", tuner_128.results_summary(1))
    print("Tuner 256:\n", tuner_256.results_summary(1))

    # Run on test set
    # tuner.get_best_models(1).evaluate(test)


if __name__ == "__main__":
    main()


"""
TODO:
* Review tune_bert_ffnn's trials in bert_ffnn_10, find the best ones and continue finding the optimal FFNN hps
* Pool BERT (already fine-tuned) pooled_outputs, and then push through a FFNN
* Look at different BERT Feed tokenization overlaps - pass data to Optimizer, which passes to fit method?
* Models using 5-fold cross validation
"""

