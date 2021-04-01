from flask import request, jsonify

from services import AbstractService
from services.user_profiler.models.model_handlers import TweetFeedModelHandler


class UserProfilerService(AbstractService):
    """ Service which profiles Twitter users """
    endpoint = "user_profiler"

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)

        # Load model handlers
        self.tweet_feed_model_handler = TweetFeedModelHandler(self.config)

    def route_profile_twitter_user_from_tweet_feed(self):
        """ Profile a Twitter user as a fake news spreader, using their tweet feed """
        username = request.args.get("username")

        feed = self.data_handler.get_twitter_api().get_user_tweet_feed(username)
        pipeline_results = self.tweet_feed_model_handler.transform(feed)
        pipeline_results["num_tweets_used"] = len(feed)
        pipeline_results["username"] = username
        return jsonify(pipeline_results), 200
