from typing import Union, Optional, Collection

import numpy as np
import tensorflow.keras as keras
from kerastuner import HyperParameters
from official.nlp.optimization import create_optimizer
from tensorflow.python.keras.callbacks import EarlyStopping

from bert import bert_layers, BertTweetFeedTokenizer
from evaluation.models import AbstractModel


class BertPooledModel(AbstractModel):
    """ User-level BERT-based fake news profiling model consisting of BertModel and PoolingModel """

    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)
        self.bert_model = BertModel(self.hyperparameters)
        self.pooling_model = PoolingModel(self.hyperparameters)

    def fit(self, x: Collection[Collection[str]], y: Collection[float]):
        self.bert_model.fit(x, y)
        x_bert_out = self.bert_model.predict(x, encoder_output=True)
        self.pooling_model.fit(x_bert_out, y)

    def predict(self, x: Collection[Collection[str]]) -> Collection[float]:
        x_bert_out = self.bert_model.predict(x, encoder_output=True)
        return self.pooling_model.predict(x_bert_out)


class BertModel(AbstractModel):
    """ Tweet-level BERT fake news profiling model, trained using raw user tweets """

    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)

        # Build BERT Keras model
        bert_input, bert_output, encoder = bert_layers(
            encoder_url=self.hyperparameters.get("BertModel_encoder_url"),
            trainable=True,
            hidden_layer_size=self.hyperparameters.get("BertModel_size"),
            return_encoder=True,
        )
        dropout = keras.layers.Dropout(self.hyperparameters.get("BertModel_dropout_rate"))(bert_output["pooled_output"])
        batch = keras.layers.BatchNormalization()(dropout)
        linear = keras.layers.Dense(
            1, activation=self.hyperparameters.get("BertModel_linear_activation"),
            kernel_regularizer=keras.regularizers.l2(self.hyperparameters.get("BertModel_linear_kernel_reg")),
            bias_regularizer=keras.regularizers.l2(self.hyperparameters.get("BertModel_linear_bias_reg")),
            activity_regularizer=keras.regularizers.l2(self.hyperparameters.get("BertModel_linear_activity_reg")),
        )(batch)

        self.tokenizer = BertTweetFeedTokenizer(encoder, self.hyperparameters.get("BertModel_size"))
        self.model = keras.Model(bert_input, linear)
        self.model_encoder_output = keras.backend.function(self.model.layers[:3], [self.model.layers[3]])

    def _compile(self, x: Collection[Collection[str]]):
        """ Compile BERT Keras model with the AdamW optimizer """
        bert_input_data_len = BertTweetFeedTokenizer.get_data_len(
            [[len(tweet) for tweet in tweet_feed] for tweet_feed in x],
            self.hyperparameters.get("BertModel_size"),
            self.hyperparameters.get("BertModel_feed_data_overlap")
        )
        num_train_steps = self.hyperparameters.get("BertModel_epochs") * bert_input_data_len // \
            self.hyperparameters.get("BertModel_batch_size")
        self.model.compile(
            optimizer=create_optimizer(
                init_lr=self.hyperparameters.get("BertModel_learning_rate"),
                num_train_steps=num_train_steps,
                num_warmup_steps=num_train_steps // 10,
                optimizer_type='adamw',
            ),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=keras.metrics.BinaryAccuracy(),
        )

    def tokenize(self,
                 x: Collection[Collection[str]],
                 y: Optional[Collection[float]] = None,
                 **kwargs) -> Union[Collection[int], Collection[float]]:
        """ Tokenize input data using BERT's tokenizer """
        x_tokenized = self.tokenizer.tokenize_input(x, **kwargs)
        if y is not None:
            y_tokenized = self.tokenizer.tokenize_labels(y)
            return x_tokenized, y_tokenized

        return x_tokenized

    def fit(self, x: Collection[Collection[str]], y: Collection[float]):
        self._compile(x)

        x_tokenized, y_tokenized = self.tokenize(x, y)
        self.model.fit(
            x_tokenized, y_tokenized,
            epochs=self.hyperparameters.get("BertModel_epochs"),
            batch_size=self.hyperparameters.get("BertModel_batch_size"),
            callbacks=[EarlyStopping("val_loss", patience=1)],
        )

    def predict(self,
                x: Collection[Collection[str]],
                encoder_output: bool = False) -> Union[Collection[float], Collection[Collection[float]]]:
        x_tokenized = self.tokenize(x)
        if encoder_output:
            return np.asarray([
                self.model_encoder_output(
                    self.tokenize([tweet_feed], overlap=self.hyperparameters.get("BertModel_feed_data_overlap"))
                ) for tweet_feed in x
            ])

        return self.model.predict(x_tokenized)


class PoolingModel(AbstractModel):
    """ User-level fake news profiling model, trained using BertModel tweet embeddings """

    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)

        # Build Pooling Keras model
        inputs = keras.layers.Input((self.hyperparameters.get("BertModel_size"),))
        dropout = keras.layers.Dropout(self.hyperparameters.get("PoolingModel_rate"))(inputs)
        batch = keras.layers.BatchNormalization()(dropout)
        linear = keras.layers.Dense(
            1, activation=self.hyperparameters.get("PoolingModel_linear_activation"),
            kernel_regularizer=keras.regularizers.l2(self.hyperparameters.get("PoolingModel_linear_kernel_reg")),
            bias_regularizer=keras.regularizers.l2(self.hyperparameters.get("PoolingModel_linear_bias_reg")),
            activity_regularizer=keras.regularizers.l2(self.hyperparameters.get("PoolingModel_linear_activity_reg")),
        )(batch)

        self.model = keras.Model(inputs, linear)

    def _compile(self, x: Collection[Collection[str]]):
        num_train_steps = self.hyperparameters.get("pooling_epochs") * len(x) // \
            self.hyperparameters.get("pooling_batch_size")
        pooling_optimizer = create_optimizer(
            init_lr=self.hyperparameters.get("pooling_learning_rate"),
            num_train_steps=num_train_steps,
            num_warmup_steps=num_train_steps // 10,
            optimizer_type='adamw',
        )
        self.model.compile(
            optimizer=pooling_optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=keras.metrics.BinaryAccuracy(),
        )

    def fit(self, x: Collection[Collection[any]], y: Collection[float]):
        self._compile(x)
        self.model.fit(
            x, y,
            epochs=self.hyperparameters.get("PoolingModel_epochs"),
            batch_size=self.hyperparameters.get("PoolingModel_batch_size"),
            callbacks=[EarlyStopping("val_loss", patience=1)],
        )

    def predict(self, x: Collection[Collection[any]]) -> Collection[float]:
        return self.model.predict(x)
