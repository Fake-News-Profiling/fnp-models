from typing import List, Callable
from functools import reduce
from joblib import load
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from models.model import AbstractDataProcessor
from models.bert_tokenizer import AbstractBertTokenizer
import services.user_profiler.models.tweet_funcs as tweetf


class BertTweetFeedDataPreprocessor(AbstractDataProcessor):
    """ Preprocess's tweet feeds to be used in the BertIndividualTweetModel """

    def __init__(self, transformers: List[Callable] = None):
        if transformers is None:
            transformers = [
                tweetf.tag_indicators,
                tweetf.replace_xml_and_html,
                tweetf.replace_emojis,
                tweetf.remove_punctuation,
                tweetf.replace_tags,
                tweetf.remove_hashtag_chars,
                tweetf.replace_accented_chars,
                tweetf.tag_numbers,
                tweetf.remove_stopwords,
                tweetf.remove_extra_spacing,
            ]
        self.transformers = transformers

    def transform(self, X):
        """ Preprocess a list of user tweets """
        return np.array([self._transform_single_tweet(tweet.text) for tweet in X])

    def _transform_single_tweet(self, tweet):
        """ Preprocess a single tweet """
        return reduce(lambda data, transformer: transformer(data), self.transformers, tweet)


class BertTweetFeedTokenizer(AbstractBertTokenizer):
    """
    Tokenizes preprocessed tweet feeds to be used in the BertIndividualTweetModel. Each
    data point is a single tokenized tweet.
    """

    def transform(self, X):
        """ Tokenize input data """
        word_ids = tf.ragged.constant([self._tokenize_single_tweet(tweet) for tweet in X])
        return self._format_bert_tokens(word_ids)

    def _tokenize_single_tweet(self, tweet):
        """ Tokenize a single tweet, truncating its tokens to model_input_size """
        tokens = self.tokenizer.tokenize(tweet)[: self.model_input_size - 2]
        return self._format_bert_word_piece_input(tokens)


class BertTweetFeedModel(AbstractDataProcessor):
    """ Holds a BERT KerasModel and loads in some pre-trained weights """

    model_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"
    model_input_size = 128

    def __init__(self, model_config):
        self.encoder = KerasLayer(self.model_url, trainable=True)
        self.model = self._load_model(model_config)

    def _load_model(self, model_config):
        # Create input layers
        def input_layer(input_name):
            return Input(shape=(self.model_input_size,), dtype=tf.int32, name=input_name)

        inputs = {
            "input_word_ids": input_layer("inputs/input_word_ids"),
            "input_mask": input_layer("inputs/input_mask"),
            "input_type_ids": input_layer("inputs/input_type_ids"),
        }

        # BERT's pooled output
        encoder_pooled_output = self.encoder(inputs)["pooled_output"]

        # Dense layer output
        dense_output = Dense(1, activation="sigmoid")(encoder_pooled_output)

        # Create the Keras model
        model = Model(inputs, dense_output)
        model.load_weights(model_config.load_path).expect_partial()
        return model

    def transform(self, X):
        prediction = self.model.predict(X)
        return prediction.reshape(1, -1)


class LogisticRegressionTweetFeedClassifier(AbstractDataProcessor):
    """ Logistic Regression classifier returning predictions and probabilities """

    def __init__(self, model_config):
        self.model = load(model_config.load_path)

    def transform(self, X):
        X_sorted = np.sort(X, axis=1)
        prediction = self.model.predict(X_sorted)[0]
        probability = self.model.predict_proba(X_sorted)[0, 1] * 100  # probability of class 1 (fake news spreader)
        return {
            "is_fake_news_spreader": bool(prediction == 1),
            "fake_news_spreader_proba": probability,
        }
