import json

import tensorflow as tf
from kerastuner import HyperParameters


def load_hyperparameters(trial_filepath: str) -> HyperParameters:
    """ Load a trials hyper-parameters from a JSON file """
    with open(trial_filepath) as trial_file:
        trial = json.load(trial_file)
        return HyperParameters.from_config(trial["hyperparameters"])


def allow_gpu_memory_growth():
    """ Enable TensorFlow GPU memory growth """
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)