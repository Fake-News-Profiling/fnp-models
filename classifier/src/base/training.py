import tensorflow as tf


def allow_gpu_memory_growth():
    """ Enable TensorFlow GPU memory growth """
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
