import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from base import AbstractModel, ScopedHyperParameters


class TfIdfModel(AbstractModel):
    """
    A basic TF-IDF model which generates TF-IDF embeddings for each users tweets, average-pools the embeddings for
    each user and then trains a LogisticRegression classifier using these embeddings
    """
    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)
        self.tfidf = TfidfVectorizer(strip_accents="unicode")
        self.model = LogisticRegression()

    def _transform(self, x):
        return np.asarray([
            np.asarray(np.mean(self.tfidf.transform(tweet_feed), axis=0)).reshape(-1,)
            for tweet_feed in x
        ])

    def fit(self, x, y):
        # Flatten data to tweet-level classification
        x_tweet_level = [tweet for tweet_feed in x for tweet in tweet_feed]

        # Fit TfidfVectorizer and transform data for user-level predictions
        self.tfidf.fit(x_tweet_level)
        x_tfidf = self._transform(x)
        self.model.fit(x_tfidf, y)

    def predict(self, x):
        x_tfidf = self._transform(x)
        return self.model.predict(x_tfidf)

    def predict_proba(self, x):
        pass
