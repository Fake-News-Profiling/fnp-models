from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from fnpmodels.models import AbstractModel


class SvmCharNGramsModel(AbstractModel):
    """ Character-level N-grams + SVM model used as a baseline in the PAN 2020 task """

    def __init__(self):
        super().__init__()
        self.tfidf = TfidfVectorizer(strip_accents="unicode", analyzer="char", ngram_range=(2, 6))
        self.model = SVC()

    @staticmethod
    def _transform(x):
        return [" ".join(tweet_feed) for tweet_feed in x]

    def train(self, x, y):
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

    def __call__(self, x, *args, **kwargs):
        return self.predict(x)
