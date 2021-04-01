from typing import Union

from kerastuner import HyperParameters

import statistical.data_extraction as ex
import statistical.data_extraction.ner.spacy
from base import ScopedHyperParameters


def get_ner_wrapper(hp: Union[HyperParameters, ScopedHyperParameters]) -> ex.AbstractNerTaggerWrapper:
    ner_library = hp.get("Ner.library")
    if ner_library == "spacy":
        return statistical.data_extraction.ner.spacy.SpacyNerTaggerWrapper(hp.get("Ner.spacy_pipeline"))
    else:
        raise ValueError("Invalid `Ner.library` name")


def get_sentiment_wrapper(hp: Union[HyperParameters, ScopedHyperParameters]) -> ex.AbstractSentimentAnalysisWrapper:
    sentiment_library = hp.get("Sentiment.library")
    if sentiment_library == "vader":
        return ex.sentiment.VaderSentimentAnalysisWrapper()
    elif sentiment_library == "stanza":
        return ex.sentiment.StanzaSentimentAnalysisWrapper()
    elif sentiment_library == "textblob":
        return ex.sentiment.TextBlobSentimentAnalysisWrapper()
    else:
        raise ValueError("Invalid `Sentiment.library` name")
