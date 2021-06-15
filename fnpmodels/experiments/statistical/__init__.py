from typing import Union

from kerastuner import HyperParameters

from fnpmodels.processing.preprocess import TweetPreprocessor, tag_indicators, replace_xml_and_html
from fnpmodels.processing.statistical.ner import AbstractNerTaggerWrapper, spacy, nltk, stanza as stanza_ner
from fnpmodels.processing.statistical.sentiment import (
    AbstractSentimentAnalysisWrapper,
    vader,
    textblob,
    stanza as stanza_sentiment,
)
from fnpmodels.experiments.experiment import AbstractSklearnExperiment
from fnpmodels.models import ScopedHyperParameters


def get_ner_wrapper(hp: Union[HyperParameters, ScopedHyperParameters]) -> AbstractNerTaggerWrapper:
    ner_library = hp.get("Ner.library")
    if ner_library == "spacy":
        return spacy.SpacyNerTaggerWrapper(hp.get("Ner.spacy_pipeline"))
    elif ner_library == "stanza":
        return stanza_ner.StanzaNerTaggerWrapper()
    elif ner_library == "nltk":
        return nltk.NltkNerTaggerWrapper()
    elif ner_library == "stanford":
        return nltk.NltkStanfordNerTaggerWrapper(hp.get("Ner.classifier_path"), hp.get("Ner.jar_path"))
    else:
        raise ValueError("Invalid `Ner.library` name")


def get_sentiment_wrapper(hp: Union[HyperParameters, ScopedHyperParameters]) -> AbstractSentimentAnalysisWrapper:
    sentiment_library = hp.get("Sentiment.library")
    if sentiment_library == "vader":
        return vader.VaderSentimentAnalysisWrapper()
    elif sentiment_library == "stanza":
        return stanza_sentiment.StanzaSentimentAnalysisWrapper()
    elif sentiment_library == "textblob":
        return textblob.TextBlobSentimentAnalysisWrapper()
    else:
        raise ValueError("Invalid `Sentiment.library` name")


class AbstractStatisticalExperiment(AbstractSklearnExperiment):

    def run(self, x, y, callbacks=None, *args, **kwargs):
        preprocessor = TweetPreprocessor([tag_indicators, replace_xml_and_html])
        x = preprocessor.transform(x)
        super().run(x, y, callbacks=callbacks, *args, **kwargs)


# For library comparisons use a default SVC
default_svc_model = {
    "Sklearn.use_pca": [True, False],
    "Sklearn.model_type": "SVC",
    "Sklearn.SVC.C": {"value": 1., "condition": {"name": "Sklearn.model_type", "value": "SVC"}},
    "Sklearn.SVC.kernel": {"value": "rbf", "condition": {"name": "Sklearn.model_type", "value": "SVC"}},
}

sklearn_models = ["LogisticRegression", "SVC", "RandomForestClassifier", "XGBClassifier"]
