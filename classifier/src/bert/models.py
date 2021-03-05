from typing import List

from tensorflow_hub import KerasLayer
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def bert_layers(encoder_url, trainable, hidden_layer_size, tokenizer_class=None, data_train=None, data_val=None,
                shuffle_data=False, tokenizer_overlap=50, return_tokenizer=False):
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

    # Tokenize input
    if tokenizer_class is not None:
        tokenizer = tokenizer_class(encoder, hidden_layer_size)
        if return_tokenizer:
            return inputs, output, tokenizer

        x_train = tokenizer.tokenize_input(data_train[0], overlap=tokenizer_overlap)
        y_train = tokenizer.tokenize_labels(data_train[1])
        x_val = tokenizer.tokenize_input(data_val[0], overlap=tokenizer_overlap)
        y_val = tokenizer.tokenize_labels(data_val[1])

        if shuffle_data:
            y_train = _shuffle_tensor_with_seed(y_train, seed=1)
            x_train = {k: _shuffle_tensor_with_seed(v, seed=1) for k, v in x_train.items()}

        return inputs, output, x_train, y_train, x_val, y_val

    return inputs, output


def _shuffle_tensor_with_seed(tensor, seed=1):
    tf.random.set_seed(1)
    return tf.random.shuffle(tensor, seed=seed)


def build_base_bert(*args, **kwargs):
    inputs, output, data = bert_layers(*args, **kwargs)
    model = Model(inputs, output["pooled_output"])

    if isinstance(data, List):
        return (model, *data)
    elif data is not None:
        return model, data

    return model
