from dataclasses import dataclass

from twython import Twython


@dataclass
class Tweet:
    """ Class to hold the contents of a Tweet """
    username: str
    text: str
    id: str


class TwitterHandler:
    """ Handles all communication with the Twitter API """

    def __init__(self, api_key: str, api_secret: str):
        self.twitter = Twython(api_key, api_secret)

    def get_user_timeline(self, username, num_tweets, min_tweet_len):
        """ Fetch tweets from the users timeline """
        raw_timeline = self.twitter.get_user_timeline(
            screen_name=username, count=400, tweet_mode="extended"
        )
        tweets = list(
            filter(
                lambda tweet: len(tweet.text) > min_tweet_len,
                map(lambda tweet: self._extract_tweet_contents(username, tweet), raw_timeline),
            )
        )

        return tweets[:num_tweets]

    def _extract_tweet_contents(self, username, tweet):
        """ Extract the contents of a raw tweet """
        return Tweet(
            username=username,
            text=tweet["full_text"],
            id=tweet["id"],
        )
