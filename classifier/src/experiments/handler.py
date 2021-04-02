import sys
import os
import json
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data import parse_dataset
from experiments.experiment import AbstractExperiment, ExperimentConfig, AbstractTfExperiment
from experiments.data_visualisation import plot_averaged_experiment_data

Experiment = Tuple[AbstractExperiment.__class__, Union[str, dict, ExperimentConfig]]


class ExperimentHandler:
    """ Handler class to run a batch of experiments """

    def __init__(self, experiments: Union[Experiment, List[Experiment]]):
        if not isinstance(experiments, list):
            experiments = [experiments]

        self.experiments = experiments

    def run_experiments(self,
                        dataset_dir: Optional[str] = None,
                        x: Optional[Union[List, np.ndarray]] = None,
                        y: Optional[Union[List, np.ndarray]] = None):
        if x is None:
            print("Loading dataset")
            x, y = parse_dataset(dataset_dir, "en")

        print("Running all experiments")
        for experiment_cls, experiment_config in self.experiments:
            experiment = self._load_experiment(experiment_cls, experiment_config)
            print("Running experiment: %s in directory %s" %
                  (experiment.config.experiment_name, experiment.config.experiment_dir))
            experiment.run(x, y)
            print("Finished running experiment %s" % experiment.config.experiment_name)

    def print_results(self, num_trials: int = 10):
        """ Print out the top `num_trials` trial results for each experiment """
        for experiment_cls, experiment_config in self.experiments:
            experiment = self._load_experiment(experiment_cls, experiment_config, save_config=False)
            df = self._build_best_trials_df(experiment, num_trials)
            print(("\n\nExperiment: %s/%s\n" + df.to_markdown()) %
                  (experiment.config.experiment_dir, experiment.config.experiment_name))

    def plot_results(self):
        """ Plot performance graphs for each TensorFlow experiment """
        for i in range(len(self.experiments)):
            self.plot_experiment(i)

    def plot_experiment_i(self, experiment_index: int, **kwargs):
        self.plot_experiment(self.experiments[experiment_index], **kwargs)

    @classmethod
    def plot_experiment(cls, experiment: Tuple[AbstractExperiment.__class__, str], **kwargs):
        experiment_cls, experiment_config = experiment

        if issubclass(experiment_cls, AbstractTfExperiment):
            experiment = cls._load_experiment_config(experiment_config, save_config=False)
            plot_averaged_experiment_data(os.path.join(experiment.experiment_dir, experiment.experiment_name), **kwargs)

    def _load_experiment(self,
                         experiment_cls: AbstractExperiment.__class__,
                         experiment_config: Union[str, dict, ExperimentConfig],
                         **kwargs) -> AbstractExperiment:
        config = self._load_experiment_config(experiment_config, **kwargs)
        return experiment_cls(config)

    @staticmethod
    def _build_best_trials_df(experiment: AbstractExperiment.__class__, num_trials: int) -> pd.DataFrame:
        """ Get the top `num_trials` trial results, and return them in a pandas DataFrame """
        results = defaultdict(list)
        for trial in experiment.tuner.oracle.get_best_trials(num_trials):
            results["trial_id"].append(trial.trial_id)
            results["Sklearn.model_type"].append(trial.hyperparameters.get("Sklearn.model_type"))
            for score in trial.metrics.metrics.keys():
                results["loss" if score == "score" else score].append(trial.metrics.get_best_value(score))

            results["parameters"].append(trial.hyperparameters.values)

        return pd.DataFrame(results)

    @staticmethod
    def _load_experiment_config(config: Union[str, dict, ExperimentConfig],
                                save_config: bool = True) -> ExperimentConfig:
        """
        Loads the experiment config to an ExperimentConfig object, and saves it to the experiment directory
        """
        if isinstance(config, str):
            if not config.endswith(".json"):
                config = os.path.join(config, "experiment_config.json")

            # Load config from filepath
            with open(config, "r") as file:
                config = json.load(file)

        if not isinstance(config, ExperimentConfig):
            # Verify it is a valid config
            config = ExperimentConfig(**config)

        # Save config to experiment directory
        if save_config:
            config_dir_path = os.path.join(config.experiment_dir, config.experiment_name)
            config_filepath = os.path.join(config_dir_path, "experiment_config.json")
            if not os.path.exists(config_dir_path):
                os.makedirs(config_dir_path, exist_ok=True)
            if os.path.exists(config_filepath):
                answer = input(
                    f"{config_filepath} already exists. Use this, overwrite, or exit? (use, overwrite, exit) ")
                if answer == "use":
                    with open(config_filepath, "r") as file:
                        return ExperimentConfig(**json.load(file))
                elif answer != "overwrite":
                    sys.exit(0)

            with open(config_filepath, "w") as file:
                config_dict = config.__dict__.copy()
                if config_dict["hyperparameters"] is None:
                    config_dict.pop("hyperparameters")

                json.dump(config_dict, file, indent=2, sort_keys=True)

        return config
