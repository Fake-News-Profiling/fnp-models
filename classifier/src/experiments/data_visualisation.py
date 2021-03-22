import json
import os
import re
from typing import Callable, Generator
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from kerastuner import HyperParameters
from tensorboard.backend.event_processing import event_multiplexer

from base import load_hyperparameters


def plot_averaged_experiment_data(experiment_dir: str,
                                  trial_label_generator: Callable[[str, HyperParameters], str] = None,
                                  trial_aggregator: Callable[[HyperParameters], str] = None,
                                  aggregation_type: str = "mean",
                                  trial_filterer: Callable[[str, HyperParameters], bool] = None):
    """
    Plot experiment data, where each trials cross-validation executions has been averaged.

    :param experiment_dir: Directory of the experiment being plotted
    :param trial_label_generator: A function which, given the trial name and hyperparameters, returns a label for
    that trial
    :param trial_aggregator: A function which generates a string from a trials hyperparameters. Trials with matching
    strings will have their metrics/results averaged in the graph (1 line plotted for all of them). By default, trials
    with the exact same hyperparameters are aggregated/combined
    :param aggregation_type: mean (default), median, max, or min
    :param trial_filterer: A function which filters out trials, depending on their hyperparameters
    """

    fig, axes = plt.subplots(nrows=2, ncols=2)
    experiment_data = defaultdict(partial(defaultdict, list))

    # Collect data from each trial in the experiment
    for trial_name in get_experiment_trials(experiment_dir):
        trial_filepath = os.path.join(experiment_dir, "trial_"+trial_name)
        trial_data_path = os.path.join(trial_filepath, "average_trial_data.json")

        if not os.path.exists(trial_data_path):
            # Average the trials cross-validation executions if not already done
            average_experiment_data(experiment_dir)

        with open(trial_data_path) as file:
            data = json.load(file)

        hyperparameters = load_hyperparameters(os.path.join(trial_filepath, "trial.json"))
        data["hyperparameters"] = hyperparameters
        data["trial_name"] = trial_name

        # Aggregate trials
        aggregated_trial_data = experiment_data[
            str(hyperparameters.values) if trial_aggregator is None else trial_aggregator(hyperparameters)]

        for key, values in data.items():
            if isinstance(values, list):
                for step, value in enumerate(values):
                    ls = aggregated_trial_data[key]
                    if step >= len(ls):
                        ls.append([value])
                    else:
                        ls[step].append(value)
            else:
                aggregated_trial_data[key] = values

    # Average metrics across aggregated trials
    aggregated_experiment_data = defaultdict(dict)
    for k, v in experiment_data.items():
        aggregated_trial_data = aggregated_experiment_data[k]

        for k2, v2 in v.items():
            if isinstance(v2, list):
                aggregator = {
                    "mean": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "median": np.median,
                }[aggregation_type]
                aggregated_trial_data[k2] = list(map(aggregator, v2))
            else:
                aggregated_trial_data[k2] = v2

    # Plot data
    for data in aggregated_experiment_data.values():
        trial_name = data["trial_name"]

        # Filter out trials
        if trial_filterer is None or not trial_filterer(trial_name, data["hyperparameters"]):
            def plot(ax, y, y_label):
                ax.plot(range(1, len(y)+1), y,
                        label=trial_name if trial_label_generator is None else
                        trial_label_generator(trial_name, data["hyperparameters"]))
                ax.set_xlabel("Epochs")
                ax.set_ylabel(y_label)

            plot(axes[0][0], data["train-epoch_loss"], "train-epoch_loss")
            plot(axes[0][1], data["validation-epoch_loss"], "validation-epoch_loss")
            plot(axes[1][0], data["train-epoch_binary_accuracy"], "train-epoch_binary_accuracy")
            plot(axes[1][1], data["validation-epoch_binary_accuracy"], "validation-epoch_binary_accuracy")

    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)


def average_experiment_data(experiment_dir: str):
    """ Get averaged epoch data for each trial in this experiment """
    for trial_name in get_experiment_trials(experiment_dir):
        average_trial_data(experiment_dir, trial_name)


def get_experiment_trials(experiment_dir: str) -> Generator[str, None, None]:
    for file in os.listdir(experiment_dir):
        match = re.match(r"^trial_(.*)$", file)
        if match is not None:
            yield match.groups()[0]


def average_trial_data(experiment_dir: str, trial_name: str):
    """ Create a JSON of averaged epoch data for this trial """

    log_filepath = os.path.join(experiment_dir, "logs", trial_name)
    trial_filepath = os.path.join(experiment_dir, "trial_"+trial_name)

    runs = {}
    for execution in os.listdir(log_filepath):
        execution_filepath = os.path.join(log_filepath, execution)

        for split in os.listdir(execution_filepath):
            if re.match("train|validation", str(split)) is not None:
                split_filepath = os.path.join(execution_filepath, split)

                for file in os.listdir(split_filepath):
                    if re.match(r"^events\.out\.tfevents\..*\.v2$", str(file)) is not None:
                        runs[f"{execution}-{split}"] = os.path.join(split_filepath, file)

    em = event_multiplexer.EventMultiplexer(runs)
    em.Reload()

    scalars_across_epochs = defaultdict(list)
    for name in runs.keys():
        run_acc = em.GetAccumulator(name)
        run_acc.Reload()

        for scalar_tag in run_acc.Tags()["scalars"]:  # [epoch_loss, epoch_binary_accuracy]
            scalars = run_acc.Scalars(scalar_tag)
            scalars.sort(key=lambda s: s.step)  # Sort by step (epoch)

            for i, step in enumerate(scalars):
                sae = scalars_across_epochs[f"{name.split('-')[-1]}-{scalar_tag}"]
                if i >= len(sae):
                    sae.append([step.value])
                else:
                    sae[i].append(step.value)

    # Average across scalars_across_epochs
    averages = {}  # {"<train/validation>-<scalar>": [<value_on_epoch_1>, <value_on_epoch_2>, ...]}
    for k, v in scalars_across_epochs.items():
        v_filtered = filter(lambda a: len(a) == len(v[0]), v)  # Remove epochs with early-stopping
        averages[k] = list(map(np.mean, v_filtered))

    # Write out to trial directory
    with open(os.path.join(trial_filepath, "average_trial_data.json"), "w") as file:
        json.dump(averages, file, indent=2, sort_keys=True)
