import copy
from collections import defaultdict

import numpy as np
import tensorflow as tf
from kerastuner import BayesianOptimization
from kerastuner.engine.tuner_utils import TunerCallback
from kerastuner.tuners import Sklearn
from sklearn.model_selection import StratifiedKFold


"""
This module contains adapted KerasTuner Tuner classes, which have been updated to use cross-validation.

The 'BayesianOptimizationCV' class uses code taken from:
* kerastuner.engine.multi_execution_tuner.MultiExecutionTuner
* kerastuner.tuners.sklearn_tuner.Sklearn

The 'SklearnCV' class uses code taken from:
* kerastuner.tuners.sklearn_tuner.Sklearn
"""


class TunerCV:
    def __init__(self, cv, preprocess=None):
        self.cv = cv
        self.cv_data = None
        self.preprocess = preprocess

    def fit_data(self, x_train, y_train, transformer_wrapper):
        """
        Splits the data into `n_splits` folds, transforms data using the transformer_wrapper, and then saves the data
        for cross-validation
        """
        self.cv_data = [transformer_wrapper(*cv_fold) for cv_fold in kfold_split_wrapper(self.cv, x_train, y_train)]


class BayesianOptimizationCV(BayesianOptimization, TunerCV):
    """ BayesianOptimization keras Tuner which performs Cross Validation """
    def __init__(self, n_splits=5, preprocess=None, *args, **kwargs):
        BayesianOptimization.__init__(self, *args, **kwargs)
        TunerCV.__init__(self, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1),
                         preprocess=preprocess)
        self.x_train_size = None

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """ A hybrid method, between Sklearn Tuner and MultiExecutionTuner (using code from both) """
        # Use epochs and batch_size from trial.hyperparameters
        fit_kwargs.setdefault("batch_size", trial.hyperparameters.get("batch_size"))
        fit_kwargs.setdefault("epochs", trial.hyperparameters.get("epochs"))

        # Create a ModelCheckpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)
        original_callbacks = fit_kwargs.pop('callbacks', [])
        metrics = defaultdict(list)

        split_data = self.cv_data if self.cv_data is not None else kfold_split_wrapper(
            self.cv, fit_kwargs["x"], fit_kwargs["y"])
        for split_num, (x_train, y_train, x_test, y_test) in enumerate(split_data):
            _, x_train, y_train, x_test, y_test = self.preprocess(
                trial.hyperparameters, x_train, y_train, x_test, y_test)

            # Copy kwargs and add data to it
            copied_fit_kwargs = copy.copy(fit_kwargs)
            copied_fit_kwargs["x"] = x_train
            copied_fit_kwargs["y"] = y_train
            copied_fit_kwargs["validation_data"] = (x_test, y_test)

            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, split_num)
            callbacks.append(TunerCallback(self, trial))
            callbacks.append(model_checkpoint)
            copied_fit_kwargs['callbacks'] = callbacks

            history = self._build_and_fit_model(trial, fit_args, copied_fit_kwargs)

            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across K-folds and send to the Oracle
        averaged_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        self.oracle.update_trial(trial.trial_id, metrics=averaged_metrics, step=self._reported_step)


class SklearnCV(Sklearn, TunerCV):
    def __init__(self, preprocess=None, *args, **kwargs):
        Sklearn.__init__(self, *args, **kwargs)
        TunerCV.__init__(self, cv=self.cv, preprocess=preprocess)

    def run_trial(self, trial, X, y, sample_weight=None, groups=None):
        """
        A modified run_trial method which can use fitted CV data
        """
        metrics = defaultdict(list)

        split_data = self.cv_data if self.cv_data is not None else kfold_split_wrapper(self.cv, X, y)
        for split_num, (x_train, y_train, x_test, y_test) in enumerate(split_data):
            if self.preprocess is not None:
                x_train, y_train, x_test, y_test = self.preprocess(trial.hyperparameters, x_train, y_train, x_test,
                                                                   y_test)

            # Build and train model
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train)

            # Evaluation the model
            if self.scoring is None:
                score = model.score(x_test, y_test)
            else:
                score = self.scoring(model, x_test, y_test)
            metrics['score'].append(score)

            if self.metrics:
                y_test_pred = model.predict(x_test)
                for metric in self.metrics:
                    result = metric(y_test, y_test_pred)
                    metrics[metric.__name__].append(result)

        trial_metrics = {name: np.mean(values) for name, values in metrics.items()}
        self.oracle.update_trial(trial.trial_id, trial_metrics)
        self.save_model(trial.trial_id, model)


def kfold_split_wrapper(kfold, x, y):
    """ Wraps Sklearn KFold.split() operations to return data instead of indices """
    splits = kfold.split(x, y)
    for train_indices, test_indices in splits:
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        yield x_array[train_indices], y_array[train_indices], x_array[test_indices], y_array[test_indices]