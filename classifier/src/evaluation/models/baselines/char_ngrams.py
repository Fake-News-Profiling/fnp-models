from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from base import AbstractModel, ScopedHyperParameters


class SvmCharNGramsModel(AbstractModel):
    """

    """
    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)
        self.tfidf = TfidfVectorizer(strip_accents="unicode", analyzer="char", ngram_range=(2, 6))
        self.model = SVC()

    @staticmethod
    def _transform(x):
        return [" ".join(tweet_feed) for tweet_feed in x]

    def fit(self, x, y):
        # Concatenate user tweets
        x = self._transform(x)
        x = self.tfidf.fit_transform(x)
        self.model.fit(x, y)

    def predict(self, x):
        x = self._transform(x)
        x = self.tfidf.transform(x)
        return self.model.predict(x)

    def predict_proba(self, x):
        pass
