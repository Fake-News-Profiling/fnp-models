import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from twython import Twython
from fnpmodels.processing.parse import parse_author_tweets


@dataclass
class Tweet:
    """ Stores the contents of a Twitter 'tweet' """
    username: str
    text: str
    id: str


@dataclass
class TwitterHandlerConfig:
    """ A configuration file for a TwitterHandler instance """

    twitter_dir: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


class AbstractTwitterHandler(ABC):
    """ Handles all communication with Twitter """

    def __init__(self, config: TwitterHandlerConfig):
        self.config = config

    @abstractmethod
    def get_user_tweet_feed(self, username: str, *args, **kwargs) -> List[Tweet]:
        """ Fetch tweets from the users timeline """
        pass


class TwitterApiHandler(AbstractTwitterHandler):
    """ Communicates with the Twitter API """

    def __init__(self, config: TwitterHandlerConfig):
        super().__init__(config)
        self.twitter = Twython(self.config.api_key, self.config.api_secret)

    def get_user_tweet_feed(self, username: str, num_tweets: int = 100, min_tweet_len: int = 10) -> List[Tweet]:
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
                "Only found %d tweets for this user, this is less than the number of tweets required (%d)" %
                (len(tweets), num_tweets)
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


class TwitterDatabaseHandler(AbstractTwitterHandler):
    """ Communicates with a local database of Twitter data """

    def get_user_tweet_feed(self, username: str, *args, **kwargs) -> List[Tweet]:
        user_path = os.path.join(self.config.twitter_dir, "tweet_feeds", username + ".xml")
        raw_tweets = list(parse_author_tweets([user_path]).values())[0]
        tweets = [Tweet(username, tweet, str(i)) for i, tweet in enumerate(raw_tweets)]
        return tweets
