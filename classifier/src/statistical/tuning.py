from collections import defaultdict
import math

import pandas as pd
import numpy as np

from data import load_data, parse_labels_to_floats, BertTweetPreprocessor
from data.preprocess import tag_indicators, replace_xml_and_html
import statistical.tune_statistical as tune


def get_best_trials(tuner, num_trials=5):
    """ Get the top `num_trials` trial results, and return them in a pandas DataFrame """
    results = defaultdict(list)
    for trial in tuner.oracle.get_best_trials(num_trials):
        results["trial_id"].append(trial.trial_id)
        results["sklearn_model"].append(trial.hyperparameters.get("sklearn_model"))
        for score in trial.metrics.metrics.keys():
            results["loss" if score == "score" else score].append(trial.metrics.get_best_value(score))

        results["parameters"].append(trial.hyperparameters.values)

    return pd.DataFrame(results)


def print_results(project_num, projects=None, num_trials=10):
    if projects is None:
        projects = ["readability", "ner", "sentiment", "combined", "combined_nn"]

    for project in projects:
        tuner = tune.sklearn_tuner(f"{project}_{project_num}")
        print(f"\n{tuner.project_name} summary:\n", get_best_trials(tuner, num_trials=num_trials).to_markdown(), sep="")


def main():

    # Load and preprocess data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data(
        "../../datasets/en_split_data.npy")

    print("Preprocessing data")
    # Remove HTML/XML tags
    preprocessor = BertTweetPreprocessor([tag_indicators, replace_xml_and_html])
    tweet_train = preprocessor.transform(tweet_train)
    tweet_val = preprocessor.transform(tweet_val)
    tweet_test = preprocessor.transform(tweet_test)

    label_train = parse_labels_to_floats(label_train)
    label_val = parse_labels_to_floats(label_val)
    label_test = parse_labels_to_floats(label_test)

    # Combine train and val as we're using Cross-Validation
    x_train = np.concatenate([tweet_train, tweet_val])
    y_train = np.concatenate([label_train, label_val])

    # Tune BERT 128
    print("Tuning models")
    project = "2"
    max_trials = 200
    tune.tune_readability_model(x_train, y_train, project, max_trials=max_trials)
    tune.tune_ner_model(x_train, y_train, project, max_trials=max_trials)
    tune.tune_sentiment_model(x_train, y_train, project, max_trials=max_trials)
    tune.tune_combined_statistical_models(x_train, y_train, project, max_trials=max_trials)
    tune.tune_combined_statistical_models(x_train, y_train, project, tune_sklearn_models=False, max_trials=max_trials)

    print_results(project)


if __name__ == "__main__":
    print_results(2)
    # main()
