from collections import defaultdict
import copy

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from kerastuner.engine.tuner_utils import TunerCallback
from kerastuner.tuners import BayesianOptimization


""" 
Academic Integrity Statement:
The 'CVBayesianOptimization.run_trial()' method uses code taken from:
* kerastuner.engine.multi_execution_tuner.MultiExecutionTuner
* kerastuner.tuners.sklearn_tuner.Sklearn
"""


class CVBayesianOptimization(BayesianOptimization):
    """ BayesianOptimization keras Tuner which performs Cross Validation """
    def __init__(self, data_preprocessing_func=None, n_splits=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_preprocessing_func = data_preprocessing_func
        self.cv_data = None
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def fit_cv_data(self, cv_data):
        """ Fit the Optimizer with pre-computed cross-validation data """
        self.cv_data = cv_data

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """ A hybrid method, between Sklearn Tuner and MultiExecutionTuner (using code from both) """
        # Create a ModelCheckpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)
        original_callbacks = fit_kwargs.pop('callbacks', [])
        metrics = defaultdict(list)

        split_data = self.cv_data if self.cv_data is not None else self.kfold_split_wrapper(
            fit_kwargs["x"], fit_kwargs["y"])
        for split_num, (x_train, y_train, x_test, y_test) in enumerate(split_data):
            if self.data_preprocessing_func is not None:
                x_train, y_train, x_test, y_test = self.data_preprocessing_func(
                    trial.hyperparameters, x_train, y_train, x_test, y_test)

            # Copy kwargs and add data to it
            copied_fit_kwargs = copy.copy(fit_kwargs)
            copied_fit_kwargs["x"] = x_train
            copied_fit_kwargs["y"] = y_train
            copied_fit_kwargs["validation_data"] = (x_test, y_test)
            trial.hyperparameters.Fixed("input_data_len", len(x_train))

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

    def kfold_split_wrapper(self, x, y):
        """ Wraps Sklearn KFold.split() operations to also be able to split BERT dictionary x data """
        if isinstance(x, dict):
            splits = self.kfold.split(list(x.values())[0], y)
            for train_indices, test_indices in splits:
                x_train = {}
                x_test = {}
                for k, v in x.items():
                    v_array = np.asarray(v)
                    x_train[k] = v_array[train_indices]
                    x_test[k] = v_array[test_indices]

                y_array = np.asarray(y)
                yield x_train, y_array[train_indices], x_test, y_array[test_indices]
        else:
            splits = self.kfold.split(x, y)
            for train_indices, test_indices in splits:
                x_array = np.asarray(x)
                y_array = np.asarray(y)
                yield x_array[train_indices], y_array[train_indices], x_array[test_indices], y_array[test_indices]
