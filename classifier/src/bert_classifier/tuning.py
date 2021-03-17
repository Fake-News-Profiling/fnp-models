from kerastuner import Objective
from kerastuner.tuners.bayesian import BayesianOptimizationOracle as BayesianOptimizationOracle
from sklearn.metrics import make_scorer, log_loss, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from bert_classifier.cv_tuners import SklearnCV


def sklearn_classifier(project_name, directory, hypermodel, hyperparameters=None, max_trials=30, preprocess=None):
    return SklearnCV(
        preprocess=preprocess,
        oracle=BayesianOptimizationOracle(
            objective=Objective("score", "min"),  # minimise log loss
            max_trials=max_trials,
            hyperparameters=hyperparameters,
        ),
        hypermodel=hypermodel,
        scoring=make_scorer(log_loss, needs_proba=True),
        metrics=[accuracy_score, f1_score],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3),
        directory=directory,
        project_name=project_name,
    )