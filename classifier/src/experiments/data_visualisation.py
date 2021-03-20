import json
import os
import re
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_multiplexer

from base import load_hyperparameters


def plot_average_experiment_data(experiment_dir, trial_label_generator=None, average_same_trials=False):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    experiment_data = defaultdict(partial(defaultdict, list))

    # Collect data from trials
    for trial_name in get_experiment_trials(experiment_dir):
        trial_filepath = os.path.join(experiment_dir, "trial_"+trial_name)
        trial_data_path = os.path.join(trial_filepath, "average_trial_data.json")
        if not os.path.exists(trial_data_path):
            average_experiment_data(experiment_dir)
        with open(trial_data_path) as file:
            data = json.load(file)

        hyperparameters = load_hyperparameters(os.path.join(trial_filepath, "trial.json"))
        data["hyperparameters"] = hyperparameters
        data["trial_name"] = trial_name

        if average_same_trials:
            same_trial_data = experiment_data[str(hyperparameters.values)]
            for key, values in data.items():
                if isinstance(values, list):
                    for step, value in enumerate(values):
                        ls = same_trial_data[key]
                        if step >= len(ls):
                            ls.append([value])
                        else:
                            ls[step].append(value)
                else:
                    same_trial_data[key] = values
        else:
            experiment_data[trial_name] = data

    if average_same_trials:
        # Average metrics across trials with the same hyperparameters
        average_data_same_trials = defaultdict(dict)
        for k, v in experiment_data.items():
            same_trial_data = average_data_same_trials[k]

            for k2, v2 in v.items():
                if isinstance(v2, list):
                    same_trial_data[k2] = list(map(np.mean, v2))
                else:
                    same_trial_data[k2] = v2

        experiment_data = average_data_same_trials

    # Plot data
    for data in experiment_data.values():
        trial_name = data["trial_name"]

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
    return fig, axes


def average_experiment_data(experiment_dir):
    """ Get averaged epoch data for each trial in this experiment """
    for trial_name in get_experiment_trials(experiment_dir):
        average_trial_data(experiment_dir, trial_name)


def get_experiment_trials(experiment_dir):
    for file in os.listdir(experiment_dir):
        match = re.match(r"^trial_(.*)$", file)
        if match is not None:
            yield match.groups()[0]


def average_trial_data(experiment_dir, trial_name):
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
        averages[k] = list(map(np.mean, v))

    # Write out to trial directory
    with open(os.path.join(trial_filepath, "average_trial_data.json"), "w") as file:
        json.dump(averages, file, indent=2, sort_keys=True)
