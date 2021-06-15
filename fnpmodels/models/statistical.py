from fnpmodels.experiments.statistical import get_ner_wrapper, get_sentiment_wrapper
from fnpmodels.processing.statistical import readability, ner, sentiment, combined_tweet_extractor
from . import ScopedHyperParameters, AbstractModel
from .sklearn import SklearnModel


class StatisticalModel(AbstractModel):
    """
    User-level statistical fake news profiling model, which uses readability, named-entity recognition, and
    sentiment features to make predictions
    """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)

        data_type = self.hp["data_type"]
        if data_type == "readability":
            self.extractor = readability.readability_tweet_extractor()
        elif data_type == "ner":
            self.extractor = ner.ner_tweet_extractor(get_ner_wrapper(self.hp))
        elif data_type == "sentiment":
            self.extractor = sentiment.sentiment_tweet_extractor(get_sentiment_wrapper(self.hp))
        elif data_type == "combined":
            self.extractor = combined_tweet_extractor(
                get_ner_wrapper(self.hp), get_sentiment_wrapper(self.hp))
        else:
            raise ValueError("Invalid value in hyperparameters for 'data_type'")

        self.model = SklearnModel(self.hp)

    def train(self, x, y):
        xt = self.extractor(x)
        self.model.train(xt, y)

    def predict(self, x):
        xt = self.extractor(x)
        return self.model.predict(xt)

    def predict_proba(self, x):
        xt = self.extractor(x)
        return self.model.predict_proba(xt)

    def __call__(self, x, *args, **kwargs):
        return self.predict_proba(x)
