import os

import tensorflow as tf

from fnpmodels.models import AbstractModel, ScopedHyperParameters
from fnpmodels.models.sklearn import SklearnModel
from .models import build_bert_model, bert_tweet_preprocessor
from .tokenize import BertIndividualTweetTokenizer, bert_tokenizer, tokenize_x, tokenize_y


class BertModel(AbstractModel):
    """ BERT tweet feed classifier """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)

        self.preprocessor = bert_tweet_preprocessor(self.hp)
        self.tokenizer = bert_tokenizer(
            self.hp["Bert.encoder_url"], self.hp["Bert.hidden_size"], BertIndividualTweetTokenizer)
        self.bert = build_bert_model(self.hp)
        self.bert_pooled = None
        self.classifier = SklearnModel(self.hp["Classifier"])

        if "Bert.weights_path" in self.hp and os.path.exists(self.hp["Bert.weights_path"] + ".index"):
            print("Loading model weights:", self.hp["Bert.weights_path"])
            self.bert.load_weights(self.hp["Bert.weights_path"])
            self.bert_pooled = self._build_bert_pooled()

    def _build_bert_pooled(self):
        return tf.keras.Model(self.bert.inputs, self.bert.layers[-2].output)

    def __call__(self, x, *args, **kwargs):
        xp = self.preprocessor(x)
        xtp = self._pool_bert_predictions(xp)
        return self.classifier(xtp)

    def _pool_bert_predictions(self, x_train):
        x_train = tf.convert_to_tensor([
            self.bert_pooled.predict(
                tokenize_x(self.hp, self.tokenizer, [tweet_feed])
            ) for tweet_feed in x_train
        ])  # (num_users, TWEET_FEED_LEN, 1, -1)
        x_train = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[3]))

        pooler = self.hp.get("Classifier.pooler")
        if pooler == "concat":
            return tf.keras.layers.Flatten()(x_train)
        elif pooler == "max":
            return tf.keras.layers.GlobalMaxPool1D()(x_train)
        elif pooler == "average":
            return tf.keras.layers.GlobalAveragePooling1D()(x_train)
        else:
            raise ValueError("Invalid value for `Classifier.pooler`")

    def train(self, x, y):
        xp = self.preprocessor(x)

        # Train BERT
        xt = tokenize_x(self.hp, self.tokenizer, xp, shuffle=True)
        yt = tokenize_y(self.hp, y)
        self.bert.fit(x=xt, y=yt, batch_size=self.hp.get("batch_size"), epochs=self.hp.get("epochs"))
        self.bert_pooled = self._build_bert_pooled()

        if "Bert.weights_path" in self.hp:
            self.bert.save_weights(self.hp["Bert.weights_path"])

        # Train pooling model
        xtp = self._pool_bert_predictions(xp)
        self.classifier.train(xtp, y)
