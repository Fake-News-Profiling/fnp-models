from kerastuner import HyperParameters

import statistical.data_extraction as ex


def get_ner_wrapper(hp: HyperParameters) -> ex.AbstractNerTaggerWrapper:
    ner_library = hp.get("Ner.library")
    if ner_library == "spacy":
        return ex.named_entity.SpacyNerTaggerWrapper(hp.get("Ner.spacy_pipeline"))
    else:
        raise ValueError("Invalid `Ner.library` name")


def get_sentiment_wrapper(hp: HyperParameters) -> ex.AbstractSentimentAnalysisWrapper:
    sentiment_library = hp.get("Sentiment.library")
    if sentiment_library == "vader":
        return ex.sentiment.VaderSentimentAnalysisWrapper()
    elif sentiment_library == "stanza":
        return ex.sentiment.StanzaSentimentAnalysisWrapper()
    elif sentiment_library == "textblob":
        return ex.sentiment.TextBlobSentimentAnalysisWrapper()
    else:
        raise ValueError("Invalid `Sentiment.library` name")
