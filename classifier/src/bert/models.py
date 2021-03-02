from typing import Type

from tensorflow_hub import KerasLayer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from bert import AbstractBertTokenizer


class BaseBertWrapper:
    """ Base wrapper for BERT KerasModels """
    def __init__(self, encoder_url: str, trainable: bool, tokenizer_class: Type[AbstractBertTokenizer], hidden_layer_size: int):
        self.encoder = KerasLayer(encoder_url, trainable=trainable)
        self.tokenizer = tokenizer_class(self.encoder, hidden_layer_size)
        self.hidden_layer_size = hidden_layer_size
        self.bert_input, self.bert_output = self._setup_bert_layers()

    def _setup_bert_layers(self):
        def input_layer(input_name):
            return Input(shape=(self.hidden_layer_size,), dtype=tf.int32, name=input_name)

        inputs = {
            'input_word_ids': input_layer("inputs/input_word_ids"),
            'input_mask': input_layer("inputs/input_mask"),
            'input_type_ids': input_layer("inputs/input_type_ids"),
        }

        # BERT's input and output layers
        return inputs, self.encoder(inputs)

    def setup_model(self):
        self.model = Model(self.bert_input, self.bert_output["pooled_output"])

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def compile(self, *args, **kwargs):
        pass

    def fit(self, X, y, batch_size, epochs, X_val, y_val, callbacks=None):
        """ Tokenizes the inputted data and fits the KerasModel """
        X_tokenized = self.tokenizer.tokenize_input(X)
        y_tokenized = self.tokenizer.tokenize_labels(y)
        X_val_tokenized = self.tokenizer.tokenize_input(X_val)
        y_val_tokenized = self.tokenizer.tokenize_labels(y_val)

        return self.model.fit(
            x=X_tokenized,
            y=y_tokenized,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_val_tokenized, y_val_tokenized)
        )

    def predict(self, X, y=None):
        """ Tokenizes the inputted data and predicts using the KerasModel """
        X_tokenized = self.tokenizer.tokenize_input(X)
        prediction = self.model.predict(X_tokenized)
        if y is not None:
            return prediction, self.tokenizer.tokenize_labels(y)

        return prediction


class SingleDenseBertWrapper(BaseBertWrapper):
    """ BERT wrapper whose output is a single Dense sigmoid layer """
    def setup_model(self):
        dense_output = Dense(1, activation='sigmoid')(self.bert_output['pooled_output'])
        self.model = Model(self.bert_input, dense_output)
