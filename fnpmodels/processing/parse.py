import os

from dataclasses import dataclass
from xml.etree import ElementTree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Tweet:
    """ Class to hold the contents of a Tweet """
    username: str
    text: str
    id: str


def parse_author_tweets(xml_filepaths):
    """Returns a dictionary of authors to a list of their tweets"""
    author_tweets = {}
    for filepath in xml_filepaths:
        xml_tree = ElementTree.parse(filepath)
        documents = xml_tree.getroot()[0]
        file_path_components = filepath.split("\\")
        file = file_path_components[len(file_path_components) - 1]

        author_id = file[0:len(file) - 4]
        tweets = [document.text for document in documents]
        author_tweets[author_id] = tweets

    return author_tweets


def _parse_author_truths(truth_filepath):
    """Returns a dictionary of authors to their truth values 1/0"""
    author_truths = {}
    with open(truth_filepath, "r") as file:
        lines = file.readlines()
        for line in lines:
            author_id, truth = line.rstrip().split(":::")
            author_truths[author_id] = truth

    return author_truths


def _filter_files(datasets_path, files, file_type):
    filtered = filter(lambda f: f.endswith(file_type), files)
    return list(map(lambda f: os.path.join(datasets_path, f), filtered))


def parse_dataset(datasets_path, language, to_pandas=False, to_raw_strings=True, training=True):
    """
    Keyword arguments:
    datasets_path -- path to the datasets directory
    language -- the language dataset to use, either "en" or "es"

    If to_pandas=True then returns pandas DataFrame, where each row contains an author id, truth value, and tweets
    1 to 100. Else returns an array of tweet feeds
    """
    language_path = os.path.join(datasets_path, language)

    # Load test authors
    with open(os.path.join(language_path, "test_authors.txt"), "r") as file:
        test_authors = set(file.read().splitlines())

    # Get each file in the directory and filter by .xml and .txt extensions
    files = os.listdir(language_path)
    xml_filepaths = _filter_files(language_path, files, ".xml")
    truth_filepath = os.path.join(language_path, "truth.txt")

    # Parse the files.
    author_tweets = parse_author_tweets(xml_filepaths)
    author_truths = _parse_author_truths(truth_filepath)

    if to_pandas:
        # Convert to a pandas DataFrame
        data = []
        for author_id, tweet_feed in author_tweets.items():
            if training and author_id in test_authors:
                continue

            d = {"author_id": author_id, "truth_value": author_truths[author_id]}
            for i, tweet in enumerate(tweet_feed, start=1):
                d["tweet_" + str(i)] = tweet

            data.append(d)

        return pd.DataFrame(data)
    else:
        # Convert to Tweet data objects
        tweet_data = []
        label_data = []
        for author_id, tweet_feed in author_tweets.items():
            if training and author_id in test_authors:
                continue

            tweet_data.append([tweet if to_raw_strings else Tweet(author_id, tweet, str(i))
                               for i, tweet in enumerate(tweet_feed, start=1)])
            label_data.append(author_truths[author_id])

        label_data = parse_labels_to_floats(label_data)

        return np.asarray(tweet_data), np.asarray(label_data)


def split_dataset(tweet_data, label_data, test_size=0.15, val_size=0.15):
    """ Takes a tweet feed dataset """
    tweet_train, tweet_other, label_train, label_other = train_test_split(
        tweet_data, label_data, test_size=(test_size + val_size),
        random_state=0,
        shuffle=True,
        stratify=label_data,
    )
    tweet_val, tweet_test, label_val, label_test = train_test_split(
        tweet_other, label_other, test_size=(test_size / (test_size + val_size)),
        random_state=42,
        shuffle=True,
        stratify=label_other,
    )

    return tweet_train, label_train, tweet_val, label_val, tweet_test, label_test


def load_data(filepath="../datasets/en_split_data.npy"):
    """ Load the saved dataset split """
    return np.load(filepath, allow_pickle=True)


def parse_labels_to_floats(y):
    return np.asarray([int(v) for v in y], dtype=np.float32)
