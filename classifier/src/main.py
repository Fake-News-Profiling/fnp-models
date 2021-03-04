import tensorflow as tf

from data import load_data, BertTweetFeedDataPreprocessor
from bert_classifier.tune_ffnn import tune_ffnn
from bert_classifier.tune_bert_ffnn import tune_bert_ffnn

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


def main():

    # Load and preprocess data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data()

    print("Preprocessing data")
    tweet_preprocessor = BertTweetFeedDataPreprocessor()
    tweet_train_processed = tweet_preprocessor.transform(tweet_train)
    tweet_val_processed = tweet_preprocessor.transform(tweet_val)
    tweet_test_processed = tweet_preprocessor.transform(tweet_test)

    print("Tuning model")
    # tuner = tune_ffnn(tweet_train_processed, label_train, tweet_val_processed, label_val)
    tuner = tune_bert_ffnn(tweet_train_processed, label_train, tweet_val_processed, label_val)


if __name__ == "__main__":
    main()
