from services import ServiceConfig
from models.model import AbstractDataProcessor, ModelPipeline
from services.user_profiler.models.bert_model import (
    BertTweetFeedDataPreprocessor,
    BertTweetFeedTokenizer,
    BertTweetFeedModel,
    LogisticRegressionTweetFeedClassifier,
)


class TweetFeedModelHandler(AbstractDataProcessor):
    """ Handles all pipelines for the tweet feed model """

    def __init__(self, service_config: ServiceConfig):
        # Load model pipelines
        self.bert_tweet_feed_pipeline = self._load_bert_tweet_feed_pipeline(service_config)

    def transform(self, X):
        result = {self.bert_tweet_feed_pipeline.name: self.bert_tweet_feed_pipeline.transform(X)}
        return result

    @staticmethod
    def _load_bert_tweet_feed_pipeline(service_config: ServiceConfig) -> ModelPipeline:
        """ Builds a ModelPipeline for the BERT tweet feed classifier"""
        bert_model = BertTweetFeedModel(service_config.models["bert_tweet_feed"])
        return ModelPipeline(
            "bert_tweet_feed",
            [
                BertTweetFeedDataPreprocessor(),
                BertTweetFeedTokenizer(bert_model.encoder, bert_model.model_input_size),
                bert_model,
                LogisticRegressionTweetFeedClassifier(service_config.models["lr_tweet_feed"]),
            ],
        )
