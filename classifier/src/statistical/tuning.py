from collections import defaultdict
import math

import pandas as pd
import numpy as np

from data import load_data, parse_labels_to_floats, BertTweetPreprocessor
from data.preprocess import tag_indicators, replace_xml_and_html
import statistical.tune_statistical as tune
from statistical.data_extraction import readability_tweet_extractor, sentiment_tweet_extractor, ner_tweet_extractor


def get_best_trials(tuner, num_trials=10):
    """ Get the top `num_trials` trial results, and return them in a pandas DataFrame """
    results = defaultdict(list)
    for trial in tuner.oracle.get_best_trials(num_trials):
        results["trial_id"].append(trial.trial_id)
        results["sklearn_model"].append(trial.hyperparameters.get("sklearn_model"))
        for score in trial.metrics.metrics.keys():
            results["loss" if score == "score" else score].append(trial.metrics.get_best_value(score))

        results["parameters"].append(trial.hyperparameters.values)

    return pd.DataFrame(results)


def get_tuner(project_num, project):
    project_name = f"{project}_{project_num}"
    return tune.nn_tuner(project_name) if "nn" in project else tune.sklearn_tuner(project_name)


def print_results(project_num, projects=None, num_trials=10):
    if projects is None:
        projects = ["readability", "ner", "sentiment", "combined", "tweet_level", "tweet_level_ensemble"]

    for project in projects:
        tuner = get_tuner(project_num, project)
        print(f"\n{tuner.project_name} summary:\n", get_best_trials(tuner, num_trials=num_trials).to_markdown(), sep="")


def get_best_models(tuner, features, num_trials=10):
    results = defaultdict(list)

    for model, trial in zip(tuner.get_best_models(num_trials), tuner.oracle.get_best_trials(num_trials)):
        results["trial_id"].append(trial.trial_id)
        results["sklearn_model"].append(trial.hyperparameters.get("sklearn_model"))
        for name, importance in zip(features, model.steps[-1][1].feature_importances_):
            results[name].append(importance)

    return pd.DataFrame(results)


def print_feature_importance(project_num, projects=None, num_trials=10):
    if projects is None:
        projects = ["readability", "ner", "sentiment", "combined"]

    for project in projects:
        tuner = get_tuner(project_num, project)
        extractor = {
            "readability": [readability_tweet_extractor()],
            "ner": [ner_tweet_extractor()],
            "sentiment": [sentiment_tweet_extractor()],
            "combined": [readability_tweet_extractor(), ner_tweet_extractor(), sentiment_tweet_extractor()],
        }[project]
        features = [feature for ex in extractor for feature in ex.feature_names]
        print(f"\n{tuner.project_name} summary:\n", get_best_models(tuner, features, num_trials).to_markdown(), sep="")


def main():
    # Load and preprocess data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data(
        "../../datasets/en_split_data.npy")

    tweet_train = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_train]
    tweet_val = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_val]
    tweet_test = [[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_test]

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
    project = "5"
    max_trials = 100
    # tune.tune_readability_model(x_train, y_train, project, max_trials=max_trials)
    # tune.tune_ner_model(x_train, y_train, project, max_trials=max_trials)
    # tune.tune_sentiment_model(x_train, y_train, project, max_trials=max_trials)
    # tune.tune_combined_statistical_models(x_train, y_train, project, max_trials=max_trials)
    # tune.tune_combined_statistical_models(x_train, y_train, project, tune_sklearn_models=False, max_trials=max_trials)
    tune.tune_tweet_level_model(x_train, y_train, project, max_trials=max_trials)

    print_results(project)
    print_feature_importance(project)


if __name__ == "__main__":
    # print_results(4)
    # print_feature_importance(4)
    main()
