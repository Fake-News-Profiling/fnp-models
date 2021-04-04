import sys

from experiments.experiment import AbstractSklearnExperiment, ExperimentConfig
import statistical.data_extraction as ex
from experiments.handler import ExperimentHandler
from experiments.statistical import get_sentiment_wrapper


""" Experiments for Sentiment Analysis models """


class SentimentExperiment(AbstractSklearnExperiment):
    """ Extract sentiment data at the user-level, and use this to train an Sklearn model """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, num_cv_splits=10)

    def input_data_transformer(self, x):
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)
        extractor = ex.sentiment_tweet_extractor(sentiment_wrapper)
        return extractor.transform(x)


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            SentimentExperiment,
            {
                "experiment_dir": "../training/statistical/sentiment",
                "experiment_name": "vader",
                "max_trials": 100,
                "hyperparameters": {"Sentiment.library": "vader"}
            }
        ), (
            SentimentExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "textblob",
                "max_trials": 100,
                "hyperparameters": {"Sentiment.library": "textblob"}
            }
        )
    ]
    handler = ExperimentHandler(experiments)
    handler.run_experiments(dataset_dir)
    handler.print_results(20)
