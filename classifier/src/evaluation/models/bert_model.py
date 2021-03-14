import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from official.nlp.optimization import AdamWeightDecay

import data.preprocess as pre
from base import AbstractModel, ScopedHyperParameters
from bert import bert_layers, BertTweetFeedTokenizer, BertIndividualTweetTokenizer
from evaluation.models.statistical import StatisticalModel
from evaluation.models.ensemble import EnsembleModel
from evaluation.models.sklearn import SklearnModel
from statistical.data_extraction import tweet_level_extractor


class BertPooledModel(AbstractModel):
    """
    User-level BERT-based fake news profiling model. This model trains BERT on individual tweets to generate
    BERT-based tweet-embeddings, it then combines these with statistical information about each tweet. The resulting
    tweet-embeddings are pooled together into user-level embeddings, and then used to profile each user.
    """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)
        self.bert_model = BertModel(self.hyperparameters.get_scope("BertModel"))
        self.pooling_model = PoolingModel(self.hyperparameters.get_scope("PoolingModel"))

        self.stats_preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
        self.stats_extractor = tweet_level_extractor()

    def extract_stats(self, x):
        x_processed = self.stats_preprocessor.transform(x)
        return np.asarray([self.stats_extractor.transform(tweet_feed) for tweet_feed in x_processed])

    def to_embeddings(self, x):
        x_bert = self.bert_model.predict(x)
        x_stats = self.extract_stats(x)
        return np.concatenate([x_bert, x_stats], axis=-1)

    def fit(self, x, y):
        # Fit BERT
        self.bert_model.fit(x, y)

        # Combine BERT and stats embeddings
        x_embedding = self.to_embeddings(x)

        # Train the pooling model
        self.pooling_model.fit(x_embedding, y)

    def predict(self, x):
        # Combine BERT and stats embeddings and predict a user-level classification
        x_embedding = self.to_embeddings(x)
        return self.pooling_model.predict(x_embedding)

    def predict_proba(self, x):
        x_embedding = self.to_embeddings(x)
        return self.pooling_model.predict_proba(x_embedding)


class BertModel(AbstractModel):
    """ Tweet-level BERT fake news profiling model, trained using user tweets """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)
        self.model = self.tokenizer = self.encoder_model = None
        self.preprocessor = pre.BertTweetPreprocessor(
            pre.select_preprocess_option(self.hyperparameters.get("preprocessing")))

    def build(self):
        """ Build and compile the BERT Keras model """
        # Build model
        bert_input, bert_output, encoder = bert_layers(
            encoder_url=self.hyperparameters.get("encoder_url"),
            trainable=True,
            hidden_layer_size=self.hyperparameters.get("size"),
            return_encoder=True,
        )
        dropout = keras.layers.Dropout(self.hyperparameters.get("dropout_rate"))(bert_output["pooled_output"])
        batch = keras.layers.BatchNormalization()(dropout)
        linear = keras.layers.Dense(
            1, activation=self.hyperparameters.get("dense_activation"),
            kernel_regularizer=keras.regularizers.l2(self.hyperparameters.get("dense_kernel_reg")),
            bias_regularizer=keras.regularizers.l2(self.hyperparameters.get("dense_bias_reg")),
            activity_regularizer=keras.regularizers.l2(self.hyperparameters.get("dense_activity_reg")),
        )(batch)

        self.tokenizer = BertIndividualTweetTokenizer(encoder, self.hyperparameters.get("size"))
        self.model = keras.Model(bert_input, linear)
        if self.hyperparameters.get("encoder_output"):
            self.encoder_model = keras.Model(self.model.inputs, self.model.layers[3].output["pooled_output"])

        # Compile model
        self.model.compile(
            optimizer=AdamWeightDecay(learning_rate=self.hyperparameters.get("learning_rate")),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=tf.metrics.BinaryAccuracy(),
        )

    def tokenize(self, x, y=None, **kwargs):
        """ Tokenize input data using BERT's tokenizer """
        x_tokenized = self.tokenizer.tokenize_input(x, **kwargs)
        if y is not None:
            y_tokenized = self.tokenizer.tokenize_labels(y)
            return x_tokenized, y_tokenized

        return x_tokenized

    def fit(self, x, y):
        self.build()
        x_processed = self.preprocessor.transform(x)
        x_tokenized, y_tokenized = self.tokenize(x_processed, y)
        self.model.fit(
            x_tokenized, y_tokenized,
            epochs=self.hyperparameters.get("epochs"),
            batch_size=self.hyperparameters.get("batch_size"),
        )

    def predict(self, x):
        x_processed = self.preprocessor.transform(x)
        if self.hyperparameters.get("encoder_output"):
            return np.asarray([
                self.encoder_model.predict(
                    self.tokenize([tweet_feed], overlap=self.hyperparameters.get("feed_data_overlap"))
                ) for tweet_feed in x_processed
            ])

        x_tokenized = self.tokenize(x_processed)
        return self.model.predict(x_tokenized)

    def predict_proba(self, x):
        x_processed = self.preprocessor.transform(x)
        x_tokenized = self.tokenize(x_processed)
        return self.model.predict(x_tokenized)


class PoolingModel(AbstractModel):
    """
    User-level fake news profiling model, which pools together tweet-level embeddings of each user and then
    classifies the user using an Sklearn model
    """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)
        self.sklearn_model = SklearnModel(hyperparameters.get("SklearnModel"))

    def pool_embeddings(self, x):
        return list({
            "max": (np.max(tweet_feed, axis=0) for tweet_feed in x),
            "average": (np.mean(tweet_feed, axis=0) for tweet_feed in x),
            "concatenate": (np.concatenate(tweet_feed) for tweet_feed in x),
        }[self.hyperparameters.get("pooling_type")])

    def fit(self, x, y):
        x_pooled = self.pool_embeddings(x)
        self.sklearn_model.fit(x_pooled, y)

    def predict(self, x):
        x_pooled = self.pool_embeddings(x)
        return self.sklearn_model.predict(x_pooled)

    def predict_proba(self, x):
        x_pooled = self.pool_embeddings(x)
        return self.sklearn_model.predict_proba(x_pooled)


def ensemble_bert_pooled_model(hyperparameters: ScopedHyperParameters):
    """ Combines the BERT Pooled model with statistical models to generate user classifications """
    return EnsembleModel(hyperparameters, models=[
        (BertPooledModel, "BertPooledModel"),
        (StatisticalModel, "ReadabilityModel"),
        (StatisticalModel, "NerModel"),
        (StatisticalModel, "SentimentModel"),
    ])
