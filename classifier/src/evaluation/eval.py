from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

import evaluation.models as models
from base import ScopedHyperParameters
from base.training import allow_gpu_memory_growth
from data import load_data, parse_labels_to_floats, parse_dataset

allow_gpu_memory_growth()


def evaluate(model, x, y, eval_metrics):
    """ Evaluate the performance of a model, returning a dict of metric results """
    predictions = model.predict(x)
    return {name: fn(y, predictions) for name, fn in eval_metrics}


def evaluate_models(x, y, eval_models, eval_metrics, cv_splits=5, save_filepath=None):
    """ Evaluate fake news profiling models, using K-fold Cross-Validation """
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True)

    # Fit and evaluate models using Cross-Validation on the training set
    cv_metrics = defaultdict(list)
    for i, (train_indices, val_indices) in enumerate(kfold.split(x, y)):
        for name, model_class, hyperparameters in eval_models:
            cv_metrics["CV split"].append(i + 1)
            model = model_class(hyperparameters)
            cv_metrics["Model"].append(name)
            model.fit(x[train_indices], y[train_indices])
            metrics = evaluate(model, x[val_indices], y[val_indices], eval_metrics)

            for metric_name, value in metrics.items():
                cv_metrics[metric_name].append(float(value))

            print(pd.DataFrame(cv_metrics).to_markdown())

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

    if save_filepath is not None:
        with open(save_filepath, "a") as file:
            file.write("\n")
            cv_df.to_markdown(file)


def main():
    # Load data
    print("Loading data")
    x, y = parse_dataset("../datasets", "en", training=False)

    # Evaluate models
    print("Beginning model evaluation")
    eval_models = [
        # Baselines
        ("RandomModel", models.baselines.RandomModel, None),
        ("TfIdfModel", models.baselines.TfIdfModel, None),
        ("SvmCharNGramsModel", models.baselines.SvmCharNGramsModel, None),
        ("Buda20NgramEnsembleModel", models.baselines.Buda20NgramEnsembleModel, None),

        # Statistical models
        ("ReadabilityStatisticalModel", models.StatisticalModel,
         ScopedHyperParameters.from_json("evaluation/models/hyperparameters/readability_model.json")),
        ("NerStatisticalModel", models.StatisticalModel,
         ScopedHyperParameters.from_json("evaluation/models/hyperparameters/ner_model.json")),
        ("SentimentStatisticalModel", models.StatisticalModel,
         ScopedHyperParameters.from_json("evaluation/models/hyperparameters/sentiment_model.json")),

        # BERT-based models
        # ("BertPooledModel", models.BertPooledModel,
        #  load_hyperparameters("models/hyperparameters/bert_model.json", to_scoped_hyperparameters=True)),
        # ("EnsembleBertPooledModel", models.ensemble_bert_pooled_model,
        #  load_hyperparameters("models/hyperparameters/ensemble_bert_model.json", to_scoped_hyperparameters=True)),
        # ("Bert256PooledModel", models.BertPooledModel,
        #  load_hyperparameters("models/hyperparameters/bert_model_256.json", to_scoped_hyperparameters=True)),
        # ("EnsembleBert256PooledModel", models.ensemble_bert_pooled_model,
        #  load_hyperparameters("models/hyperparameters/ensemble_bert_model_256.json", to_scoped_hyperparameters=True)),
        # ("EnsembleBertPooledModelFfnnOut", models.ensemble_bert_pooled_model,
        #  load_hyperparameters("models/hyperparameters/ensemble_bert_model_ffnn_out.json",
        #                       to_scoped_hyperparameters=True)),
        # ("BertPooledModelFfnnOut", models.BertPooledModel,
        #  load_hyperparameters("models/hyperparameters/bert_model_ffnn_out.json", to_scoped_hyperparameters=True)),
        # ("Bert256PooledModelFfnnOut", models.BertPooledModel,
        #  load_hyperparameters("models/hyperparameters/bert_model_256_ffnn_out.json", to_scoped_hyperparameters=True)),
        # ("EnsembleBert256PooledModelFfnnOut", models.ensemble_bert_pooled_model,
        #  load_hyperparameters("models/hyperparameters/ensemble_bert_model_256_ffnn_out.json",
        #                       to_scoped_hyperparameters=True)),
    ]
    metrics = [
        ("Loss", tf.keras.losses.binary_crossentropy),
        ("Accuracy", tf.metrics.binary_accuracy),
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1", f1_score),
    ]
    with tf.device("/gpu:0"):
        evaluate_models(x, y, eval_models, metrics, save_filepath="evaluation/eval_results.txt")
        evaluate_models(x, y, eval_models, metrics, save_filepath="evaluation/eval_results.txt")
        evaluate_models(x, y, eval_models, metrics, save_filepath="evaluation/eval_results.txt")


if __name__ == "__main__":
    main()
