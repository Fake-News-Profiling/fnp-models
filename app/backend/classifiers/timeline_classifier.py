import logging

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert import tokenization
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

logging.getLogger("tensorflow").setLevel(logging.ERROR)


class BertModel:
    def __init__(self, encoder_url, bert_input_size, encoding):
        # Get BERT encoder and tokenizer
        self.encoder = hub.KerasLayer(encoder_url, trainable=True)
        self.bert_input_size = bert_input_size
        self.encoding = encoding
        self.tokenizer = tokenization.FullTokenizer(
            self.encoder.resolved_object.vocab_file.asset_path.numpy(),
            do_lower_case=self.encoder.resolved_object.do_lower_case.numpy(),
        )

    def build(self):
        inputs = self.compile_inputs()

        # Get the BERT pooled output
        encoder_pooled_output = self.encoder(inputs)["pooled_output"]

        # Get the Dense layer output
        dense_output = Dense(1, activation="sigmoid")(encoder_pooled_output)

        # Create the Keras model and compile
        self.model = Model(inputs, dense_output)

    def compile_inputs(self):
        # Create BERT Input layers
        def input_layer(input_name):
            return Input(shape=(self.bert_input_size,), dtype=tf.int32, name=input_name)

        return dict(
            input_word_ids=input_layer("inputs/input_word_ids"),
            input_mask=input_layer("inputs/input_mask"),
            input_type_ids=input_layer("inputs/input_type_ids"),
        )

    def set_encoding(self, new_encoding):
        self.encoding = new_encoding

    def predict(self, x):
        if self.encoding == "individual":
            # TODO - unsure if this is correct.
            x_encoded = self.encode_input(x)
            return self.model.predict(x_encoded)

        elif self.encoding == "feed":
            result = []
            for tweet_feed in x:
                x_encoded = self.encode_input([tweet_feed])
                prediction = self.model.predict(x_encoded)
                result.append(prediction)

            # Returns (n, k, 768) array, where k is the number of chunks (variable),
            # n is the number of tweet feeds (input size), and 768 is BERT's hidden layer size.
            return np.asarray(result)

    def encode_input(self, x):  # x should always be a list or array!
        # Encode tweets
        def encode_tweet(tweet):
            tokens = self.tokenizer.tokenize(tweet)[: self.bert_input_size - 2]
            tokens.append("[SEP]")
            tokens.insert(0, "[CLS]")
            return self.tokenizer.convert_tokens_to_ids(tokens)

        # Encode tweet feed into chunks
        def encode_tweet_feed(tweet_feed):
            feed_tokens = self.tokenizer.tokenize(tweet_feed)
            tokens = [
                feed_tokens[i : i + self.bert_input_size - 2]
                for i in range(0, len(feed_tokens), self.bert_input_size - 52)
            ]
            #             tokens[-1] = feed_tokens[len(feed_tokens)-self.bert_input_size+2:len(feed_tokens)] # Remove? - fills the last chunk

            def encode_tokens(tokens):
                tokens.append("[SEP]")
                tokens.insert(0, "[CLS]")
                return self.tokenizer.convert_tokens_to_ids(tokens)

            return list(map(encode_tokens, tokens))

        if self.encoding == "individual":
            input_word_ids = tf.ragged.constant([encode_tweet(tweet) for tweet in x])

        elif self.encoding == "feed":
            encoded_tweet_feed = [encode_tweet_feed(tweet_feed) for tweet_feed in x]
            input_word_ids = tf.ragged.constant(
                [tweet for tweet_feed in encoded_tweet_feed for tweet in tweet_feed]
            )
            self.y_pattern = [len(tweet_feed) for tweet_feed in encoded_tweet_feed]

        input_mask = tf.ones_like(input_word_ids)

        # Pad word_ids and mask
        input_word_ids = input_word_ids.to_tensor()
        input_mask = input_mask.to_tensor()

        padding = tf.constant([[0,0], [0, (self.bert_input_size - input_mask.shape[1])]])
        input_word_ids = tf.pad(input_word_ids, padding, "CONSTANT")
        input_mask = tf.pad(input_mask, padding, "CONSTANT")

        input_type_ids = tf.zeros_like(input_mask)

        # Return encoded inputs
        x_encoded = dict(
            input_word_ids=input_word_ids, input_mask=input_mask, input_type_ids=input_type_ids
        )

        return x_encoded
