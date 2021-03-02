import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from official.nlp import optimization


def basic_ffnn(bert_hidden_layer_size, learning_rate, num_training_steps, warmup_rate=0.1):
    model = tf.keras.models.Sequential([
        Input(shape=(bert_hidden_layer_size,), dtype=tf.float32),
        Dense(bert_hidden_layer_size * 2, activation="relu"),
        Dense(bert_hidden_layer_size, activation="relu"),
        Dense(bert_hidden_layer_size // 2, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    optimizer = optimization.create_optimizer(
        learning_rate, num_training_steps, int(warmup_rate * num_training_steps), optimizer_type='adamw')
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
    )

    return model









