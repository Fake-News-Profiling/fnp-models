import sys
import os
import json
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data import parse_dataset
from experiments.experiment import AbstractExperiment, ExperimentConfig


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
            experiment = self._load_experiment(experiment_cls, experiment_config)
            df = self._build_best_trials_df(experiment, num_trials)
            print(("\n\nExperiment: %s/%s\n" + df.to_markdown) %
                  (experiment.config.experiment_dir, experiment.config.experiment_name))

    def _load_experiment(self,
                         experiment_cls: AbstractExperiment.__class__,
                         experiment_config: Union[str, dict, ExperimentConfig]) -> AbstractExperiment:
        config = self._load_experiment_config(experiment_config)
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
    def _load_experiment_config(config: Union[str, dict, ExperimentConfig]) -> ExperimentConfig:
        """
        Loads the experiment config to an ExperimentConfig object, and saves it to the experiment directory
        """
        if isinstance(config, str):
            # Load config from filepath
            with open(config, "r") as file:
                config = json.load(file)

        if not isinstance(config, ExperimentConfig):
            # Verify it is a valid config
            config = ExperimentConfig(**config)

        # Save config to experiment directory
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
