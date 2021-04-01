from statistical.data_extraction.readability import readability_tweet_extractor
from statistical.data_extraction.ner import ner_tweet_extractor, AbstractNerTaggerWrapper
from statistical.data_extraction.sentiment import sentiment_tweet_extractor, AbstractSentimentAnalysisWrapper
from statistical.data_extraction.tweet_level import tweet_level_extractor
from statistical.data_extraction.preprocessing import TweetStatsExtractor


def combined_tweet_extractor(ner_wrapper: AbstractNerTaggerWrapper,
                             sentiment_wrapper: AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
    """ TweetStatsExtractor which extracts readability, NER, and sentiment features """
    readability_extractor = readability_tweet_extractor()
    ner_extractor = ner_tweet_extractor(ner_wrapper)
    sentiment_extractor = sentiment_tweet_extractor(sentiment_wrapper)

    extractor = TweetStatsExtractor([
        *readability_extractor.extractors,
        *ner_extractor.extractors,
        *sentiment_extractor.extractors,
    ])
    extractor.feature_names = [
        *readability_extractor.feature_names,
        *ner_extractor.feature_names,
        *sentiment_extractor.feature_names,
    ]
    return extractor
