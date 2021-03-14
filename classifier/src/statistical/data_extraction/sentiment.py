import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import statistical.data_extraction.preprocessing as pre

analyzer = SentimentIntensityAnalyzer()


""" Sentiment model data extraction functions """


def sentiment_tweet_extractor():
    """ Create a TweetStatsExtractor for named entity recognition features """
    extractor = pre.TweetStatsExtractor(extractors=[tweet_sentiment_scores, overall_sentiment])
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


def tweet_sentiment_scores(user_tweets):
    """ Returns the average, standard deviation, and max/min sentiment scores of the user """
    tweet_polarity = [analyzer.polarity_scores(tweet)['compound'] for tweet in user_tweets]
    sent_mean = np.mean(tweet_polarity, axis=0)
    sent_std_dev = np.std(tweet_polarity)
    sent_max = np.max(tweet_polarity, axis=0)
    sent_min = np.min(tweet_polarity, axis=0)

    num_pos, num_neu, num_neg = 0, 0, 0
    for score in tweet_polarity:
        if score >= 0.05:
            num_pos += 1
        elif score <= -0.05:
            num_neg += 1
        else:
            num_neu += 1

    return np.asarray([sent_mean, sent_std_dev, sent_max, sent_min, num_pos, num_neu, num_neg])


def overall_sentiment(user_tweets):
    """ Returns the overall sentiment when all of the users tweets have been concatenated """
    return analyzer.polarity_scores(". ".join(user_tweets))['compound']
