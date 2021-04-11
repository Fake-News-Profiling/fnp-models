from typing import Union

from kerastuner import HyperParameters

from data import BertTweetPreprocessor
from data.preprocess import tag_indicators, replace_xml_and_html
import statistical.data_extraction as ex
from experiments.experiment import AbstractSklearnExperiment
from statistical.data_extraction.sentiment import vader, stanza as stanza_sentiment, textblob
from statistical.data_extraction.ner import spacy, stanza as stanza_ner, nltk
from base import ScopedHyperParameters


def get_ner_wrapper(hp: Union[HyperParameters, ScopedHyperParameters]) -> ex.AbstractNerTaggerWrapper:
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


def get_sentiment_wrapper(hp: Union[HyperParameters, ScopedHyperParameters]) -> ex.AbstractSentimentAnalysisWrapper:
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
        preprocessor = BertTweetPreprocessor([tag_indicators, replace_xml_and_html])
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
