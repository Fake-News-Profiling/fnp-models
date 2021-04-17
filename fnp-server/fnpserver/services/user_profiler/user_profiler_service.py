from typing import Tuple

from flask import request, jsonify, Response
from fnpmodels.models import ScopedHyperParameters
from fnpmodels.models.ensemble import EnsembleBertModel

from fnpserver.services import AbstractService


class UserProfilerService(AbstractService):
    """ A service which profiles Twitter users """

    def __init__(self, config_dict: dict, endpoint: str):
        super().__init__(config_dict, endpoint)

        # Load models
        ensemble_bert_model_hp = ScopedHyperParameters.from_json(
            self.config.models["ensemble_bert_model"].hyperparameters)
        self.tweet_feed_model = EnsembleBertModel(ensemble_bert_model_hp)

    def route_profile_twitter_user_from_tweet_feed(self) -> Tuple[Response, int]:
        """ Profile a Twitter user as a fake-news spreader, using their tweet feed """
        username = request.args.get("username")

        tweet_feed = self.data_handler.get_twitter_handler().get_user_tweet_feed(username)
        data = self.tweet_feed_model([tweet_feed])
        data["num_tweets_used"] = len(tweet_feed)
        data["username"] = username
        return jsonify(data), 200
