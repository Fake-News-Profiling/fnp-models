from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional

import numpy as np

import statistical.data_extraction.preprocessing as pre


""" Sentiment model data extraction functions """


@dataclass
class Sentiment:
    compound: float
    classification: str
    negative: Optional[float] = 0
    neutral: Optional[float] = 0
    positive: Optional[float] = 0


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
        partial(aggregated_compound_tweet_sentiment_scores_plus_counts, sentiment_wrapper=sentiment_wrapper),
        partial(overall_compound_sentiment_score, sentiment_wrapper=sentiment_wrapper),
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
                           sentiment_wrapper: AbstractSentimentAnalysisWrapper = None) -> np.ndarray:
    """
    Returns the average, standard deviation, and range of each polarity score, as well as counts of negative,
    neutral and positive tweets.

    Returns:
        [negative_mean, neutral_mean, positive_mean, compound_mean, negative_std, ..., compound_min, negative_max, ...,
        num_negative, num_neutral, num_positive]
    """

    def sentiment_scores_as_list(tweet):
        sentiment = sentiment_wrapper.sentiment(tweet)
        counts = [0, 0, 0]
        indices = {"negative": 0, "neutral": 1, "positive": 2}
        counts[indices[sentiment.classification]] += 1
        return sentiment.negative, sentiment.positive, sentiment.compound, *counts

    tweet_sentiments = np.asarray(list(map(sentiment_scores_as_list, tweet_feed)))
    sentiment_mean = np.mean(tweet_sentiments[:, :3], axis=0)
    sentiment_std = np.std(tweet_sentiments[:, :3], axis=0)
    sentiment_min = np.min(tweet_sentiments[:, :3], axis=0)
    sentiment_max = np.max(tweet_sentiments[:, :3], axis=0)
    num_neg = np.sum(tweet_sentiments[:, 3])
    num_neu = np.sum(tweet_sentiments[:, 4])
    num_pos = np.sum(tweet_sentiments[:, 5])
    return np.concatenate([sentiment_mean, sentiment_std, sentiment_min[-1:], sentiment_max, [num_neg, num_neu, num_pos]])


def aggregated_compound_tweet_sentiment_scores(tweet_feed: List[str],
                                               sentiment_wrapper: AbstractSentimentAnalysisWrapper = None
                                               ) -> List[float]:
    """ Returns the average, standard deviation and range of compound scores """
    scores = tweet_sentiment_scores(tweet_feed, sentiment_wrapper)
    compound_mean = scores[2]
    compound_std = scores[5]
    compound_min = scores[6]
    compound_max = scores[9]
    return [compound_mean, compound_std, compound_min, compound_max]


def aggregated_compound_tweet_sentiment_scores_plus_counts(tweet_feed: List[str],
                                                           sentiment_wrapper: AbstractSentimentAnalysisWrapper = None
                                                           ) -> List[float]:
    """ Returns the average, standard deviation and range of compound scores """
    scores = tweet_sentiment_scores(tweet_feed, sentiment_wrapper)
    compound_mean = scores[2]
    compound_std = scores[5]
    compound_min = scores[6]
    compound_max = scores[9]
    return [compound_mean, compound_std, compound_min, compound_max, *scores[-3:]]


def aggregated_tweet_sentiment_scores(tweet_feed: List[str],
                                      sentiment_wrapper: AbstractSentimentAnalysisWrapper = None) -> List[float]:
    """ Returns the average, standard deviation and range of sentiment and polarity scores """
    scores = tweet_sentiment_scores(tweet_feed, sentiment_wrapper)
    return scores[:-3]


def overall_sentiment_score(tweet_feed: List[str],
                            sentiment_wrapper: AbstractSentimentAnalysisWrapper = None) -> List[float]:
    """ Returns the overall sentiment when all of the users tweets have been concatenated """
    sentiment = sentiment_wrapper.sentiment(". ".join(tweet_feed))
    return [sentiment.negative, sentiment.neutral, sentiment.positive, sentiment.compound]


def overall_compound_sentiment_score(tweet_feed: List[str],
                                     sentiment_wrapper: AbstractSentimentAnalysisWrapper = None) -> float:
    """ Returns the overall compound sentiment when all of the users tweets have been concatenated """
    return overall_sentiment_score(tweet_feed, sentiment_wrapper)[-1]
