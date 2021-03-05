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

    # Tune BERT 128
    # tuner = tune_bert_ffnn(tweet_train_processed, label_train, tweet_val_processed, label_val,
    #                        bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
    #                        bert_size=128,
    #                        project_name="bert_ffnn_8")

    # Tune BERT 256
    tuner = tune_bert_nn_classifier(
        tweet_train_processed, label_train, tweet_val_processed, label_val,
        bert_encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        bert_size=128,
        project_name="bert_ffnn_11",
        bert_weights="../training/bert_clf/initial_eval/bert_ffnn_6/trial_0b51ec0b3a25404c4fb4ee3bdeaa8d66/checkpoints/"
                     "epoch_0/checkpoint",
        max_trials=100,
        epochs=20,
        batch_sizes=[8, 16, 24, 32, 48, 64, 80, 96],
    )
    print(tuner.results_summary(2))


if __name__ == "__main__":
    main()


"""
TODO:
* Review tune_bert_ffnn's trials in bert_ffnn_10, find the best ones and continue finding the optimal FFNN hps
* Pool BERT (already fine-tuned) pooled_outputs, and then push through a FFNN
* Look at different BERT Feed tokenization overlaps
"""

