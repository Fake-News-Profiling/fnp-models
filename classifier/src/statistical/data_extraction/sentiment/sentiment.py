from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, List

import numpy as np

import statistical.data_extraction.preprocessing as pre


""" Sentiment model data extraction functions """


@dataclass
class Sentiment:
    compound: float
    classification: str


class AbstractSentimentAnalysisWrapper(ABC):
    """ Sentiment Analysis wrapper to provide a uniform interface to sentiment analysis libraries """

    def __init__(self, analyser: Any):
        self.analyser = analyser

    @abstractmethod
    def sentiment(self, text: str) -> Sentiment:
        """ Return the sentiment scores of this text: compound, neutral, positive, and negative sentiments """
        pass


def sentiment_tweet_extractor(sentiment_wrapper: AbstractSentimentAnalysisWrapper) -> pre.TweetStatsExtractor:
    """ Create a TweetStatsExtractor for named entity recognition features """
    extractor = pre.TweetStatsExtractor([
        partial(tweet_sentiment_scores, sentiment_wrapper=sentiment_wrapper),
        partial(overall_sentiment, sentiment_wrapper=sentiment_wrapper),
    ])
    extractor.feature_names = [
        "Average tweet sentiment",
        "Standard deviation of tweet sentiments",
        "Max tweet sentiment",
        "Min tweet sentiment",
        "Number of positive tweets",
        "Number of neutral tweets",
        "Number of negative tweets",
        "Overall sentiment of the user",
    ]
    return extractor


def tweet_sentiment_scores(tweet_feed: List[str],
                           sentiment_wrapper: AbstractSentimentAnalysisWrapper = None) -> List[float]:
    """ Returns the average, standard deviation, and max/min sentiment scores of the user """
    tweet_comp_sentiments = []
    num_pos = num_neu = num_neg = 0
    for tweet in tweet_feed:
        sentiment = sentiment_wrapper.sentiment(tweet)
        tweet_comp_sentiments.append(sentiment.compound)
        if sentiment.classification == "positive":
            num_pos += 1
        elif sentiment.classification == "neutral":
            num_neu += 1
        elif sentiment.classification == "negative":
            num_neg += 1

    sent_mean = np.mean(tweet_comp_sentiments, axis=0)
    sent_std_dev = np.std(tweet_comp_sentiments)
    sent_max = np.max(tweet_comp_sentiments, axis=0)
    sent_min = np.min(tweet_comp_sentiments, axis=0)
    return [sent_mean, sent_std_dev, sent_max, sent_min, num_pos, num_neu, num_neg]


def overall_sentiment(tweet_feed: List[str],
                      sentiment_wrapper: AbstractSentimentAnalysisWrapper = None) -> float:
    """ Returns the overall sentiment when all of the users tweets have been concatenated """
    return sentiment_wrapper.sentiment(". ".join(tweet_feed)).compound
