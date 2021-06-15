from functools import partial

import numpy as np
from tensorflow_hub import KerasLayer
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from fnpmodels.processing import preprocess as pre
from fnpmodels.experiments.models import CompileOnFitKerasModel


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


class BertUserLevelClassifier(tf.keras.layers.Layer):
    """
    A BERT classifier layer, which takes in `Bert.tweet_feed_len` tweets, creates embeddings for each one using
    shared BERT weights and pools the results.
    """

    def __init__(self, hp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters = hp

        self.bert_inputs, _, self.bert_encoder = bert_layers(
            self.hyperparameters.get("Bert.encoder_url"),
            trainable=True,
            hidden_layer_size=self.hyperparameters.get("Bert.hidden_size"),
            return_encoder=True,
        )

        if self.hyperparameters.get("selected_encoder_outputs") != "default":
            self.bert_pooled_output_pooling = tf.keras.layers.Dense(
                self.hyperparameters.get("Bert.hidden_size"), activation="tanh")
        else:
            self.bert_pooled_output_pooling = None

        if self.hyperparameters.get("Bert.tweet_feed_len") > 1:
            self.pooling = self._make_pooler()

        self.dropout = tf.keras.layers.Dropout(self.hyperparameters.get("Bert.dropout_rate"))
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def _make_pooler(self):
        # Returns a pooler of type: (batch_size, TWEET_FEED_LEN, Bert.hidden_size) => (batch_size, Bert.hidden_size)
        pooler = self.hyperparameters.get("Bert.pooler")

        if pooler == "max":
            return tf.keras.layers.GlobalMaxPool1D()  # shape == (batch_size, Bert.hidden_size)
        elif pooler == "average":
            return tf.keras.layers.GlobalAveragePooling1D()  # shape == (batch_size, Bert.hidden_size)
        elif pooler == "concat":
            return tf.keras.layers.Flatten()  # shape == (batch_size, TWEET_FEED_LEN * Bert.hidden_size)
        else:
            raise ValueError("Invalid value for `Bert.pooler`")

    @tf.function
    def _run_bert_encoder(self, inputs, training=None):
        # inputs.shape == (batch_size, 3, Bert.hidden_size)
        # Returns a Tensor with shape (batch_size, 128)
        bert_input = dict(input_word_ids=inputs[:, 0], input_mask=inputs[:, 1], input_type_ids=inputs[:, 2])
        bert_out = self.bert_encoder(bert_input, training=training)

        selected_encoder_outputs = self.hyperparameters.get("selected_encoder_outputs")
        pooled_output = extract_bert_pooled_output(bert_out, self.bert_pooled_output_pooling, selected_encoder_outputs)
        return tf.reshape(pooled_output, (-1, 1, self.hyperparameters.get("Bert.hidden_size")))

    @tf.function
    def _accumulate_bert_outputs(self, inputs, training=None):
        # inputs.shape == (batch_size, TWEET_FEED_LEN, 3, Bert.hidden_size)
        # Returns a tensor with shape (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        return tf.concat([self._run_bert_encoder(
            inputs[:, i], training=training) for i in range(self.hyperparameters.get("Bert.tweet_feed_len"))], axis=1)

    def call(self, inputs, training=None, mask=None):
        # inputs.shape == (batch_size, TWEET_FEED_LEN, 3)
        # Returns a tensor with shape (batch_size, 1)
        x_train = self._accumulate_bert_outputs(inputs, training=training)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size)

        if self.hyperparameters.get("Bert.tweet_feed_len") > 1:
            x_train = self.pooling(x_train)
            # x_train.shape == (batch_size, Bert.hidden_size) or (batch_size, TWEET_FEED_LEN * Bert.hidden_size)

        if self.hyperparameters.Fixed("Bert.use_batch_norm", False):
            x_train = self.batch_norm(x_train)

        x_train = self.dropout(x_train, training=training)
        return x_train

    def get_config(self):
        pass


class BertPlusStatsUserLevelClassifier(BertUserLevelClassifier):
    """ A BertUserLevelClassifier which also pools tweet-level statistical data """

    def call(self, inputs, training=None, mask=None):
        # inputs.shape == [(batch_size, TWEET_FEED_LEN, 3, Bert.hidden_size),
        #                  (batch_size, TWEET_FEED_LEN, NUM_STATS_FEATURES)]
        # Returns a tensor with shape (batch_size, 1)
        bert_data = inputs[0]
        stats_data = inputs[1]
        x_train = self._accumulate_bert_outputs(bert_data, training=training)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size)
        x_train = tf.concat([stats_data, x_train], axis=-1)
        # x_train.shape == (batch_size, TWEET_FEED_LEN, Bert.hidden_size + NUM_STATS_FEATURES)
        x_train = self.pooling(x_train)
        # x_train.shape == (batch_size, Bert.hidden_size + NUM_STATS_FEATURES)
        x_train = self.dropout(x_train, training=training)
        return x_train


def bert_tweet_preprocessor(hp):
    preprocessing_choice = hp.get("Bert.preprocessing")

    if preprocessing_choice != "none":
        data_transformers = {
            "[remove_emojis]": [pre.remove_emojis, pre.replace_unicode, pre.replace_tags],
            "[remove_emojis, remove_punctuation]": [
                pre.remove_emojis, pre.replace_unicode, pre.remove_colons, pre.remove_punctuation_and_non_printables,
                pre.replace_tags, pre.remove_hashtags],
            "[remove_emojis, remove_tags]": [
                pre.remove_emojis, pre.replace_unicode, partial(pre.replace_tags, remove=True)],
            "[remove_emojis, remove_tags, remove_punctuation]": [
                pre.remove_emojis, pre.replace_unicode, pre.remove_colons, pre.remove_punctuation_and_non_printables,
                partial(pre.replace_tags, remove=True), pre.remove_hashtags],
            "[tag_emojis]": [partial(pre.replace_emojis, with_desc=False), pre.replace_unicode, pre.replace_tags],
            "[tag_emojis, remove_punctuation]": [
                partial(pre.replace_emojis, with_desc=False), pre.replace_unicode, pre.remove_colons,
                pre.remove_punctuation_and_non_printables, pre.replace_tags, pre.remove_hashtags],
            "[replace_emojis]": [pre.replace_emojis, pre.replace_unicode, pre.replace_tags],
            "[replace_emojis_no_sep]": [partial(pre.replace_emojis, sep=""), pre.replace_unicode, pre.replace_tags],
            "[replace_emojis_no_sep, remove_tags]": [
                partial(pre.replace_emojis, sep=""), pre.replace_unicode, partial(pre.replace_tags, remove=True)],
            "[replace_emojis_no_sep, remove_tags, remove_punctuation]": [
                partial(pre.replace_emojis, sep=""), pre.replace_unicode, pre.remove_colons,
                pre.remove_punctuation_and_non_printables, partial(pre.replace_tags, remove=True), pre.remove_hashtags],
        }[preprocessing_choice]

        # Preprocess data
        return pre.TweetPreprocessor(
            [pre.tag_indicators, pre.replace_xml_and_html] + data_transformers + [pre.remove_extra_spacing])


def build_bert_model(hp):
    # BERT tweet chunk pooler
    inputs = tf.keras.layers.Input((hp.get("Bert.tweet_feed_len"), 3, hp.get("Bert.hidden_size")), dtype=tf.int32)
    bert_outputs = BertUserLevelClassifier(hp)(inputs)
    # bert_outputs.shape == (batch_size, Bert.hidden_size)

    # Hidden feed-forward layers
    bert_output_dim = bert_outputs.shape[-1]
    for i in range(hp.Fixed("Bert.num_hidden_layers", 0)):
        dense = tf.keras.layers.Dense(
            bert_output_dim // max(1, i * 2),  # Half the dimension of each feed-forward layer
            activation=hp.get("Bert.hidden_dense_activation"),
            kernel_regularizer=tf.keras.regularizers.l2(hp.get("Bert.dense_kernel_reg")),
        )(bert_outputs)
        if hp.Fixed("Bert.use_batch_norm", False):
            dense = tf.keras.layers.BatchNormalization()(dense)
        dropout = tf.keras.layers.Dropout(hp.get("Bert.dropout_rate"))(dense)
        bert_outputs = dropout

    # Final linear layer
    linear = tf.keras.layers.Dense(
        1,
        activation=hp.Fixed("Bert.dense_activation", "linear"),
        kernel_regularizer=tf.keras.regularizers.l2(hp.get("Bert.dense_kernel_reg")),
    )(bert_outputs)

    return CompileOnFitKerasModel(inputs, linear, optimizer_learning_rate=hp.get("learning_rate"))
