import numpy as np
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


def bert_tokenizer(encoder_url, hidden_layer_size, tokenizer_class):
    encoder = KerasLayer(encoder_url, trainable=False)
    return tokenizer_class(encoder, hidden_layer_size)


def tokenize_bert_input(encoder_url, hidden_layer_size, tokenizer_class, x_train, y_train, x_val=None, y_val=None,
                        shuffle=False, feed_overlap=50, return_tokenizer=False):
    tokenizer = bert_tokenizer(encoder_url, hidden_layer_size, tokenizer_class)
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


def extract_bert_pooled_output(encoder_out, pooling_layer, pooling_strategy="default"):
    encoder_outputs = encoder_out["encoder_outputs"]

    if pooling_strategy == "default":
        return encoder_out["pooled_output"]
    else:
        encoder_outputs = encoder_outputs[{
            "first_layer": np.s_[:1],
            "2nd_to_last_hidden_layer": np.s_[-2:-1],
            "last_hidden_layer": np.s_[-1:],
            "sum_all_but_last_hidden_layers": np.s_[:-1],
            "sum_all_hidden_layers": np.s_[:],
            "sum_4_2nd_to_last_hidden_layers": np.s_[-5:-1],
            "sum_last_4_hidden_layers": np.s_[:-4],
            "concat_4_2nd_to_last_hidden_layers": np.s_[-5:-1],
            "concat_last_4_hidden_layers": np.s_[:-4],
        }[pooling_strategy]]

        # Pool `encoder_outputs` by summing or concatenating
        if pooling_strategy.startswith("concat"):
            # Concatenate layer outputs, and extract the concatenated '[CLF]' embeddings
            # pooled_outputs.shape == (batch_size, len(encoder_outputs) * hidden_size)
            pooled_output = tf.concat(encoder_outputs, axis=-1)[:, 0, :]
        else:
            # Extract the '[CLF]' embeddings of each layer, and then sum them
            pooled_outputs = tf.convert_to_tensor(encoder_outputs)[:, :, 0, :]
            # pooled_outputs.shape == (batch_size, hidden_size)
            pooled_output = tf.reduce_sum(pooled_outputs, axis=0)

        # Pass pooled_outputs through a tanh layer (as they did in the original BERT paper)
        return pooling_layer(pooled_output)
