from typing import List
from dataclasses import dataclass
from twython import Twython

from data import TwitterApiConfig


@dataclass
class Tweet:
    """ Stores the contents of a Twitter 'tweet' """
    username: str
    text: str
    id: str


class TwitterApiHandler:
    """ Handles all communication with the Twitter API """

    def __init__(self, twitter_api_config: TwitterApiConfig):
        self.twitter = Twython(twitter_api_config.api_key, twitter_api_config.api_secret)

    def get_user_tweet_feed(self, username: str, num_tweets: int = 100, min_tweet_len: int = 10) -> List[Tweet]:
        """ Fetch tweets from the users timeline """
        raw_timeline = self.twitter.get_user_timeline(
            screen_name=username, count=400, tweet_mode="extended"
        )
        tweets = list(
            filter(
                lambda tweet: len(tweet.text) > min_tweet_len,
                map(lambda tweet: self._extract_tweet_contents(username, tweet), raw_timeline),
            )
        )[:num_tweets]

        if len(tweets) < num_tweets:
            raise RuntimeError(
                f"Only found {len(tweets)} tweets for this user, this is less than the number of tweets required ({num_tweets})"
            )

        return tweets

    @staticmethod
    def _extract_tweet_contents(username: str, tweet: dict) -> Tweet:
        """ Extract the contents of a raw tweet """
        return Tweet(
            username=username,
            text=tweet["full_text"],
            id=tweet["id"],
        )
