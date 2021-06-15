from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fnpmodels.evaluation import scoring
from fnpmodels.experiments import allow_gpu_memory_growth
from fnpmodels.processing.parse import parse_dataset
from fnpmodels import models


def evaluate(model, x, y, eval_metrics):
    """ Evaluate the performance of a model, returning a dict of metric results """
    predictions = model.predict(x)
    return {name: fn(y, predictions) for name, fn in eval_metrics}


def evaluate_models(x, y, eval_models, eval_metrics, cv_splits=10, num_repetitions=5, save_filepath=None):
    """ Evaluate fake news profiling models, using K-fold Cross-Validation """

    # Perform `num_repetitions` cross-validations
    cv_metrics = defaultdict(list)
    for r in range(num_repetitions):
        kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True)

        # Fit and evaluate models using Cross-Validation on the dataset
        for i, (train_indices, val_indices) in enumerate(kfold.split(x, y)):
            for name, model_class, hyperparameters in eval_models:
                cv_metrics["Repetition"].append(r + 1)
                cv_metrics["CV split"].append(i + 1)
                model = model_class(hyperparameters)
                cv_metrics["Model"].append(name)
                model.train(x[train_indices], y[train_indices])
                metrics = evaluate(model, x[val_indices], y[val_indices], eval_metrics)

                for metric_name, value in metrics.items():
                    cv_metrics[metric_name].append(float(value))

        print(pd.DataFrame(cv_metrics).to_markdown())

    cv_df = pd.DataFrame(cv_metrics)

    # Compute metric averages
    final_df = pd.DataFrame()
    for model_name in cv_df["Model"].unique():
        def reduce_col(col_data, aggr):
            if col_data.dtype == object:
                return col_data.iloc[0]
            return aggr(col_data)

        # Compute average and std scores
        for name, agg in [("Average", np.mean), ("Standard Deviation", np.std), ("Max", np.max), ("Min", np.min)]:
            row = cv_df[cv_df["Model"] == model_name].apply(partial(reduce_col, aggr=agg))
            row.loc["Agg"] = name
            final_df = final_df.append(row, ignore_index=True)

    final_df = final_df[["Model", "Agg", *[name for name, _ in eval_metrics]]].sort_values(["Model", "Agg"])
    print(f"\n{cv_splits}-fold Cross-Validation results:\n", final_df.to_markdown(), sep="", end="\n")

    if save_filepath is not None:
        with open(save_filepath, "a") as file:
            file.write("\n")
            final_df.to_markdown(file)


def main():
    # Load data
    print("Loading data")
    x, y = parse_dataset("../../../datasets", "en", training=False)

    # Evaluate models
    print("Beginning model evaluation")
    eval_models = [
        # Baselines
        ("RandomModel", models.baselines.RandomModel, None),
        ("TfIdfModel", models.baselines.TfIdfModel, None),
        ("SvmCharNGramsModel", models.baselines.SvmCharNGramsModel, None),
        ("Buda20NgramEnsembleModel", models.baselines.Buda20NgramEnsembleModel, None),
    ]
    metrics = [
        ("Loss", tf.keras.losses.binary_crossentropy),
        ("Accuracy", tf.metrics.binary_accuracy),
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1", f1_score),
        ("ROC AUC", roc_auc_score),
        ("True positives", scoring.true_positives),
        ("False negatives", scoring.false_negatives),
        ("True negatives", scoring.true_negatives),
        ("False positives", scoring.false_positives),
    ]
    with tf.device("/gpu:0"):
        evaluate_models(x, y, eval_models, metrics, save_filepath="evaluation/evaluation_results.txt")


if __name__ == "__main__":
    allow_gpu_memory_growth()
    main()
