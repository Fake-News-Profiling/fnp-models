from typing import List

from tensorflow_hub import KerasLayer
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def bert_layers(encoder_url, trainable, hidden_layer_size, return_encoder=False):
    encoder = KerasLayer(encoder_url, trainable=trainable)

    # BERT's input and output layers
    def input_layer(input_name):
        return Input(shape=(hidden_layer_size,), dtype=tf.int32, name=input_name)

    inputs = {
        'input_word_ids': input_layer("inputs/input_word_ids"),
        'input_mask': input_layer("inputs/input_mask"),
        'input_type_ids': input_layer("inputs/input_type_ids"),
    }
    output = encoder(inputs)

    if return_encoder:
        return inputs, output, encoder

    return inputs, output


def tokenize_bert_input(encoder_url, hidden_layer_size, tokenizer_class, x_train, y_train, x_val, y_val,
                        x_test=None, y_test=None, shuffle=False, feed_overlap=50):
    encoder = KerasLayer(encoder_url, trainable=False)

    # Tokenize input
    tokenizer = tokenizer_class(encoder, hidden_layer_size)
    x_train = tokenizer.tokenize_input(x_train, overlap=feed_overlap)
    y_train = tokenizer.tokenize_labels(y_train)
    x_val = tokenizer.tokenize_input(x_val, overlap=feed_overlap)
    y_val = tokenizer.tokenize_labels(y_val)

    if shuffle:
        y_train = _shuffle_tensor_with_seed(y_train, seed=1)
        x_train = {k: _shuffle_tensor_with_seed(v, seed=1) for k, v in x_train.items()}

    if x_test is not None:
        x_test = tokenizer.tokenize_input(x_test, overlap=feed_overlap)
        y_test = tokenizer.tokenize_labels(y_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    return x_train, y_train, x_val, y_val


def _shuffle_tensor_with_seed(tensor, seed=1):
    tf.random.set_seed(1)
    return tf.random.shuffle(tensor, seed=seed)


def build_base_bert(encoder_url, trainable, hidden_layer_size, tokenizer_class):
    inputs, output, encoder = bert_layers(encoder_url, trainable, hidden_layer_size, return_encoder=True)
    tokenizer = tokenizer_class(encoder, hidden_layer_size)
    model = Model(inputs, output["pooled_output"])

    return model, tokenizer
