from statistical.data_extraction.readability import readability_tweet_extractor
from statistical.data_extraction.named_entity import ner_tweet_extractor
from statistical.data_extraction.sentiment import sentiment_tweet_extractor
from statistical.data_extraction.tweet_level import tweet_level_extractor
from statistical.data_extraction.preprocessing import TweetStatsExtractor


def combined_tweet_extractor():
    readability_extractor = readability_tweet_extractor()
    ner_extractor = ner_tweet_extractor()
    sentiment_extractor = sentiment_tweet_extractor()
    extractor = TweetStatsExtractor([
        *readability_extractor.extractors,
        *ner_extractor.extractors,
        *sentiment_extractor.extractors
    ])
    extractor.feature_names = [
        *readability_extractor.feature_names,
        *ner_extractor.feature_names,
        *sentiment_extractor.feature_names
    ]
    return extractor
