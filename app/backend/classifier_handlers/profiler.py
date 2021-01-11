from dataclasses import dataclass

from backend.api_handlers.twitter_handler import Tweet
from backend.classifiers.timeline_preprocessor import TweetPreprocessor


@dataclass
class Profile:
    """ Class to hold the results of a profiler report """

    username: str
    num_tweets_assessed: int
    spreader_probability: float
    spreader_tweet: Tweet
    spreader_tweet_probability: float


class FakeNewsProfiler:
    def __init__(self, twitter_handler, classifier_handler):
        self.twitter_handler = twitter_handler
        self.classifier_handler = classifier_handler

    def classify_user_timeline(self, username, num_tweets=100, min_tweet_len=10):
        """ Return a classification of a user, given their timeline """
        # Fetch and preprocess timeline tweets
        tweets = self.twitter_handler.get_user_timeline(username, num_tweets, min_tweet_len)
        preprocessor = TweetPreprocessor(tweets)

        # Classify the tweets
        spreader_prob = self.classifier_handler.predict_fake_news_spreader_prob(
            preprocessor.get_tweet_feed_dataset()
        )
        (spreader_tweet, spreader_tweet_prob) = self.classifier_handler.predict_tweet_with_highest_prob(
            tweets, preprocessor.get_individual_tweets_dataset()
        )

        return Profile(
            username=username,
            num_tweets_assessed=len(tweets),
            spreader_probability=spreader_prob,
            spreader_tweet=spreader_tweet,
            spreader_tweet_probability=spreader_tweet_prob,
        )
