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


def tokenize_bert_input(encoder_url, hidden_layer_size, tokenizer_class, x_train, y_train, x_val=None, y_val=None,
                        shuffle=False, feed_overlap=50, return_tokenizer=False):
    encoder = KerasLayer(encoder_url, trainable=False)

    # Tokenize input
    tokenizer = tokenizer_class(encoder, hidden_layer_size)
    x_train = tokenizer.tokenize_input(x_train, overlap=feed_overlap)
    y_train = tokenizer.tokenize_labels(y_train)

    if shuffle:
        x_train, y_train = shuffle_bert_data(x_train, y_train)

    result = [x_train, y_train]

    if x_val is not None:
        x_val = tokenizer.tokenize_input(x_val, overlap=feed_overlap)
        y_val = tokenizer.tokenize_labels(y_val) if y_val is not None else None
        result += [x_val, y_val]

    if return_tokenizer:
        result.append(tokenizer)

    return result


def shuffle_bert_data(x, y, seed=1):
    y_shuffled = _shuffle_tensor_with_seed(y, seed=seed)
    x_shuffled = {k: _shuffle_tensor_with_seed(v, seed=seed) for k, v in x.items()}
    return x_shuffled, y_shuffled


def _shuffle_tensor_with_seed(tensor, seed=1):
    tf.random.set_seed(1)
    return tf.random.shuffle(tensor, seed=seed)


def build_base_bert(encoder_url, trainable, hidden_layer_size, tokenizer_class):
    inputs, output, encoder = bert_layers(encoder_url, trainable, hidden_layer_size, return_encoder=True)
    tokenizer = tokenizer_class(encoder, hidden_layer_size)
    model = Model(inputs, output["pooled_output"])

    return model, tokenizer
