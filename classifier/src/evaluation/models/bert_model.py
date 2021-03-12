import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from kerastuner import HyperParameters
from official.nlp.optimization import create_optimizer
from tensorflow import sigmoid

from base import AbstractModel
from bert import bert_layers, BertTweetFeedTokenizer
from data import BertTweetPreprocessor


class BertPooledModel(AbstractModel):
    """ User-level BERT-based fake news profiling model consisting of BertModel and PoolingModel """

    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)
        self.preprocessor = BertTweetPreprocessor()
        self.bert_model = BertModel(self.hyperparameters)
        self.pooling_model = PoolingModel(self.hyperparameters)

    def fit(self, x, y, x_val, y_val):
        x_processed = self.preprocessor.transform(x)
        x_val_processed = self.preprocessor.transform(x_val)
        assert len(x) == len(x_processed)

        self.bert_model.fit(x_processed, y, x_val_processed, y_val)
        x_bert_out = self.bert_model.predict(x_processed, encoder_output=True)
        assert len(x) == len(x_bert_out)

        x_val_bert_out = self.bert_model.predict(x_val_processed, encoder_output=True)

        self.pooling_model.fit(x_bert_out, y, x_val_bert_out, y_val)

    def predict(self, x):
        x_processed = self.preprocessor.transform(x)
        x_bert_out = self.bert_model.predict(x_processed, encoder_output=True)
        logit_predictions = self.pooling_model.predict(x_bert_out)
        return np.round(sigmoid(logit_predictions)).reshape(-1,)


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
            1, activation=self.hyperparameters.get("BertModel_dense_activation"),
            kernel_regularizer=keras.regularizers.l2(self.hyperparameters.get("BertModel_dense_kernel_reg")),
            bias_regularizer=keras.regularizers.l2(self.hyperparameters.get("BertModel_dense_bias_reg")),
            activity_regularizer=keras.regularizers.l2(self.hyperparameters.get("BertModel_dense_activity_reg")),
        )(batch)

        self.tokenizer = BertTweetFeedTokenizer(encoder, self.hyperparameters.get("BertModel_size"))
        self.model = keras.Model(bert_input, linear)
        self.encoder_model = keras.Model(self.model.inputs, self.model.layers[3].output["pooled_output"])

    def _compile(self, x):
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

    def _tokenize(self, x, y=None, **kwargs):
        """ Tokenize input data using BERT's tokenizer """
        x_tokenized = self.tokenizer.tokenize_input(x, **kwargs)
        if y is not None:
            y_tokenized = self.tokenizer.tokenize_labels(y)
            return x_tokenized, y_tokenized

        return x_tokenized

    def fit(self, x, y, x_val, y_val):
        self._compile(x)

        x_tokenized, y_tokenized = self._tokenize(x, y)
        x_val, y_val = self._tokenize(x_val, y_val)

        self.model.fit(
            x_tokenized, y_tokenized,
            epochs=self.hyperparameters.get("BertModel_epochs"),
            batch_size=self.hyperparameters.get("BertModel_batch_size"),
            validation_data=(x_val, y_val)
        )

    def predict(self, x, encoder_output=False):
        x_tokenized = self._tokenize(x)
        if encoder_output:
            return np.asarray([
                self.encoder_model.predict(
                    self._tokenize([tweet_feed], overlap=self.hyperparameters.get("BertModel_feed_data_overlap"))
                ) for tweet_feed in x
            ])

        return self.model.predict(x_tokenized)


class PoolingModel(AbstractModel):
    """ User-level fake news profiling model, trained using BertModel tweet embeddings """

    def __init__(self, hyperparameters: HyperParameters):
        super().__init__(hyperparameters)

        # Build Pooling Keras model
        inputs = keras.layers.Input((self.hyperparameters.get("BertModel_size"),))
        dropout = keras.layers.Dropout(self.hyperparameters.get("PoolingModel_dropout_rate"))(inputs)
        batch = keras.layers.BatchNormalization()(dropout)
        linear = keras.layers.Dense(
            1, activation=self.hyperparameters.get("PoolingModel_dense_activation"),
            kernel_regularizer=keras.regularizers.l2(self.hyperparameters.get("PoolingModel_dense_kernel_reg")),
            bias_regularizer=keras.regularizers.l2(self.hyperparameters.get("PoolingModel_dense_bias_reg")),
            activity_regularizer=keras.regularizers.l2(self.hyperparameters.get("PoolingModel_dense_activity_reg")),
        )(batch)

        self.model = keras.Model(inputs, linear)

    def _compile(self, x):
        num_train_steps = self.hyperparameters.get("PoolingModel_epochs") * len(x) // \
            self.hyperparameters.get("PoolingModel_batch_size")
        pooling_optimizer = create_optimizer(
            init_lr=self.hyperparameters.get("PoolingModel_learning_rate"),
            num_train_steps=num_train_steps,
            num_warmup_steps=num_train_steps // 10,
            optimizer_type='adamw',
        )
        self.model.compile(
            optimizer=pooling_optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=keras.metrics.BinaryAccuracy(),
        )

    def _pool(self, x):
        if self.hyperparameters.get("PoolingModel_pooling_type") == "max":
            return tf.convert_to_tensor([np.max(tweet_feed, axis=0) for tweet_feed in x])
        elif self.hyperparameters.get("PoolingModel_pooling_type") == "average":
            return tf.convert_to_tensor([np.mean(tweet_feed, axis=0) for tweet_feed in x])
        else:
            raise RuntimeError("Invalid 'PoolingModel_pooling_type' value, it should be 'average' or 'max'")

    def fit(self, x, y, x_val, y_val):
        self._compile(x)
        x_pooled = self._pool(x)
        assert (len(x), self.hyperparameters.get("BertModel_size")) == x_pooled.shape

        x_val = self._pool(x_val)

        self.model.fit(
            x_pooled, y,
            epochs=self.hyperparameters.get("PoolingModel_epochs"),
            batch_size=self.hyperparameters.get("PoolingModel_batch_size"),
            validation_data=(x_val, y_val)
        )

    def predict(self, x):
        x_pooled = self._pool(x)
        return self.model.predict(x_pooled)
