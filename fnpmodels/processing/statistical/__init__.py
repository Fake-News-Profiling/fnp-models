from .extractor import TweetStatsExtractor
from . import readability, sentiment, ner


def combined_tweet_extractor(ner_wrapper: ner.AbstractNerTaggerWrapper,
                             sentiment_wrapper: sentiment.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
    """ TweetStatsExtractor which extracts readability, NER, and sentiment features """
    readability_extractor = readability.readability_tweet_extractor()
    ner_extractor = ner.ner_tweet_extractor(ner_wrapper)
    sentiment_extractor = sentiment.sentiment_tweet_extractor(sentiment_wrapper)

    extract = TweetStatsExtractor([
        *readability_extractor.extractors,
        *ner_extractor.extractors,
        *sentiment_extractor.extractors,
    ])
    extract.feature_names = [
        *readability_extractor.feature_names,
        *ner_extractor.feature_names,
        *sentiment_extractor.feature_names,
    ]
    return extract
