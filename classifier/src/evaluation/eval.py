from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

import evaluation.models as models
from base import load_hyperparameters
from base.training import allow_gpu_memory_growth
from data import load_data, parse_labels_to_floats


allow_gpu_memory_growth()


def evaluate(model, x, y, eval_metrics):
    """ Evaluate the performance of a model, returning a dict of metric results """
    predictions = model.predict(x)
    return {name: fn(y, predictions) for name, fn in eval_metrics}


def evaluate_models(x, y, x_test, y_test, eval_models, eval_metrics, cv_splits=5):
    """ Evaluate fake news profiling models, using K-fold Cross-Validation """
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=2)

    # Fit and evaluate models using Cross-Validation on the training set
    cv_metrics = defaultdict(list)
    for i, (train_indices, val_indices) in enumerate(kfold.split(x, y)):
        print("y_true", y[val_indices])

        for name, model_class, hyperparameters in eval_models:
            cv_metrics["CV split"].append(i + 1)
            model = model_class(hyperparameters)
            cv_metrics["Model"].append(name)
            model.fit(x[train_indices], y[train_indices])
            metrics = evaluate(model, x[val_indices], y[val_indices], eval_metrics)

            for metric_name, value in metrics.items():
                cv_metrics[metric_name].append(float(value))

            print(pd.DataFrame(cv_metrics).to_markdown())

            # TODO - Write model CV results to file here

    # Compute metric averages
    cv_df = pd.DataFrame(cv_metrics)
    for model_name in cv_df["Model"].unique():
        def reduce_col(col_data):
            if col_data.dtype == object:
                return col_data.iloc[0]
            return np.mean(col_data)

        row = cv_df[cv_df["Model"] == model_name].apply(reduce_col)
        row.loc["CV split"] = "Average"
        cv_df = cv_df.append(row, ignore_index=True)

    cv_df = cv_df.sort_values(["Model", "CV split"])
    print(f"\n{cv_splits}-fold Cross-Validation results:\n", cv_df.to_markdown(), sep="", end="\n")

    # Fit the models on the entire training set and evaluate them on the test set
    test_metrics = defaultdict(list)
    for name, model_class, hyperparameters in eval_models:
        model = model_class(hyperparameters)
        model.fit(x, y)
        metrics = evaluate(model, x_test, y_test, eval_metrics)
        test_metrics["Model"].append(name)

        for metric_name, value in metrics.items():
            test_metrics[metric_name].append(float(value))

    test_df = pd.DataFrame(test_metrics)
    print(f"\nFinal test set results:\n", test_df.to_markdown(), sep="", end="\n")

    with open("src/evaluation/eval_results.txt", "a") as file:
        cv_df.to_markdown(file)
        file.write("\n")
        test_df.to_markdown(file)


def main():
    # Load data
    print("Loading data")
    tweet_train, label_train, tweet_val, label_val, tweet_test, label_test = load_data(
        filepath="../../datasets/en_split_data.npy")
    x = np.asarray([[tweet.text for tweet in tweet_feed] for tweet_feed in np.concatenate([tweet_train, tweet_val])])
    y = parse_labels_to_floats(np.concatenate([label_train, label_val]))
    x_test = np.asarray([[tweet.text for tweet in tweet_feed] for tweet_feed in tweet_test])
    y_test = parse_labels_to_floats(label_test)

    # Evaluate models
    print("Beginning model evaluation")
    eval_models = [
        # Baselines
        # ("RandomModel", models.RandomModel, None),
        # ("TfIdfModel", models.TfIdfModel, None),
        # ("Buda20NgramEnsembleModel", models.Buda20NgramEnsembleModel, None),

        # Statistical models
        # ("ReadabilityStatisticalModel", models.StatisticalModel,
        #  load_hyperparameters("models/hyperparameters/readability_model.json", to_scoped_hyperparameters=True)),
        # ("NerStatisticalModel", models.StatisticalModel,
        #  load_hyperparameters("models/hyperparameters/ner_model.json", to_scoped_hyperparameters=True)),
        # ("SentimentStatisticalModel", models.StatisticalModel,
        #  load_hyperparameters("models/hyperparameters/sentiment_model.json", to_scoped_hyperparameters=True)),
        # ("CombinedStatisticalModel", models.StatisticalModel,
        #  load_hyperparameters("models/hyperparameters/combined_statistical_model.json",
        #                       to_scoped_hyperparameters=True)),
        # ("EnsembleStatisticalModel", models.ensemble_statistical_model, load_hyperparameters(
        #     "models/hyperparameters/ensemble_statistical_model.json", to_scoped_hyperparameters=True)),

        # BERT-based models
        ("BertPooledModel", models.BertPooledModel,
         load_hyperparameters("models/hyperparameters/bert_model.json", to_scoped_hyperparameters=True)),
        ("EnsembleBertPooledModel", models.ensemble_bert_pooled_model,
         load_hyperparameters("models/hyperparameters/ensemble_bert_model.json", to_scoped_hyperparameters=True)),
    ]
    metrics = [
        ("Loss", tf.keras.losses.binary_crossentropy),
        ("Accuracy", tf.metrics.binary_accuracy),
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1", f1_score),
    ]
    with tf.device("/gpu:0"):
        evaluate_models(x, y, x_test, y_test, eval_models, metrics)


if __name__ == "__main__":
    main()
