import data.preprocess as pre
from base import ScopedHyperParameters, AbstractModel
from evaluation.models.ensemble import EnsembleModel
from evaluation.models.sklearn import SklearnModel
from experiments.statistical import get_ner_wrapper, get_sentiment_wrapper
from statistical.data_extraction import readability_tweet_extractor, ner_tweet_extractor, sentiment_tweet_extractor, \
    combined_tweet_extractor


class StatisticalModel(AbstractModel):
    """
    User-level statistical fake news profiling model, which uses readability, named-entity recognition, and
    sentiment (or all 3) features to make predictions
    """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)

        # Data preprocessing and extraction
        self.preprocessor = pre.BertTweetPreprocessor([pre.tag_indicators, pre.replace_xml_and_html])
        data_type = self.hyperparameters["data_type"]

        if data_type == "readability":
            self.extractor = readability_tweet_extractor()
        elif data_type == "ner":
            self.extractor = ner_tweet_extractor(get_ner_wrapper(self.hyperparameters))
        elif data_type == "sentiment":
            self.extractor = sentiment_tweet_extractor(get_sentiment_wrapper(self.hyperparameters))
        elif data_type == "combined":
            self.extractor = combined_tweet_extractor(
                get_ner_wrapper(self.hyperparameters), get_sentiment_wrapper(self.hyperparameters))
        else:
            raise ValueError("Invalid value in hyperparameters for 'data_type'")

        self.sklearn_model = SklearnModel(self.hyperparameters["Sklearn"])

    def preprocess(self, x):
        return self.extractor.transform(
            self.preprocessor.transform(x)
        )

    def fit(self, x, y):
        xt = self.preprocess(x)
        self.sklearn_model.fit(xt, y)

    def predict(self, x):
        xt = self.preprocess(x)
        return self.sklearn_model.predict(xt)

    def predict_proba(self, x):
        xt = self.preprocess(x)
        return self.sklearn_model.predict_proba(xt)


def ensemble_statistical_model(hyperparameters: ScopedHyperParameters):
    """ Combines multiple user-level statistical models into one user classification """
    return EnsembleModel(hyperparameters, models=[
        (StatisticalModel, "ReadabilityModel"),
        (StatisticalModel, "NerModel"),
        (StatisticalModel, "SentimentModel"),
    ])
