from dacite import from_dict
import toml

from services import ServiceConfig
from services.user_profiler.models.model_handlers import TweetFeedModelHandler
from data.data_handler import DataHandler


class UserProfilerService:
    def __init__(self, path_to_config: str, data_handler: DataHandler):
        config = from_dict(ServiceConfig, toml.load(path_to_config))
        self.data_handler = data_handler

        # Load model handlers
        self.tweet_feed_model_handler = TweetFeedModelHandler(config)

    def profile_twitter_user_from_tweet_feed(self, username: str) -> dict:
        """ Profile a Twitter user as a fake news spreader, using their tweet feed """
        feed = self.data_handler.get_twitter_api().get_user_tweet_feed(username)
        pipeline_results = self.tweet_feed_model_handler.transform(feed)
        pipeline_results["num_tweets_used"] = len(feed)
        pipeline_results["username"] = username
        return pipeline_results
