from data import load_data, BertTweetFeedDataPreprocessor
from bert_classifier.tuning import tune_ffnn


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
    tuner = tune_ffnn(tweet_train_processed, label_train, tweet_val_processed, label_val)


if __name__ == "__main__":
    main()
