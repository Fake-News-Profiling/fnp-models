import sys
from functools import partial

import statistical.data_extraction.sentiment.sentiment as sent
from experiments.handler import ExperimentHandler
from experiments.statistical import get_sentiment_wrapper, AbstractStatisticalExperiment, default_svc_model
from statistical.data_extraction import TweetStatsExtractor

""" Experiments for Sentiment Analysis models """


class AbstractSentimentExperiment(AbstractStatisticalExperiment):

    def input_data_transformer(self, x):
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)
        return self.get_extractor(sentiment_wrapper).transform(x)

    @staticmethod
    def get_extractor(hp) -> TweetStatsExtractor:
        pass


class CompoundSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            lambda tweet_feed: sent.aggregated_compound_tweet_sentiment_scores_plus_counts(
                tweet_feed, sentiment_wrapper=sentiment_wrapper)[:1]
        ])
        return extractor


class SentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            lambda tweet_feed: sent.tweet_sentiment_scores(tweet_feed, sentiment_wrapper=sentiment_wrapper)[:3]
        ])
        return extractor


class AggregatedCompoundSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            lambda tweet_feed: sent.aggregated_compound_tweet_sentiment_scores_plus_counts(
                tweet_feed, sentiment_wrapper=sentiment_wrapper)[:-3]
        ])
        return extractor


class AggregatedSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            lambda tweet_feed: sent.aggregated_tweet_sentiment_scores(tweet_feed, sentiment_wrapper=sentiment_wrapper)
        ])
        return extractor


class AggregatedCompoundPlusCountsSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            partial(sent.aggregated_compound_tweet_sentiment_scores_plus_counts, sentiment_wrapper=sentiment_wrapper)
        ])
        return extractor


class AggregatedPlusCountsSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            lambda tweet_feed: sent.tweet_sentiment_scores(tweet_feed, sentiment_wrapper=sentiment_wrapper)
        ])
        return extractor


class AggregatedCompoundPlusCountsPlusOverallSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            partial(sent.aggregated_compound_tweet_sentiment_scores_plus_counts, sentiment_wrapper=sentiment_wrapper),
            partial(sent.overall_compound_sentiment_score, sentiment_wrapper=sentiment_wrapper),
        ])
        return extractor


class AggregatedPlusCountsPlusOverallSentimentExperiment(AbstractSentimentExperiment):
    """
    Extract averaged sentiment and polarity scores:
    """

    @staticmethod
    def get_extractor(sentiment_wrapper: sent.AbstractSentimentAnalysisWrapper) -> TweetStatsExtractor:
        extractor = TweetStatsExtractor([
            partial(sent.tweet_sentiment_scores, sentiment_wrapper=sentiment_wrapper),
            partial(sent.overall_compound_sentiment_score, sentiment_wrapper=sentiment_wrapper),
        ])
        extractor.feature_names = []
        return extractor


def library_comparison_handler():
    """ Comparing VADER and TextBlob libraries """
    sentiment_libraries = ["vader", "textblob"]
    experiments = [
        (
            CompoundSentimentExperiment,
            {
                "experiment_dir": "../training/statistical/sentiment/library_comparison",
                "experiment_name": library,
                "max_trials": 2,
                "num_cv_splits": 10,
                "hyperparameters": {"Sentiment.library": library, **default_svc_model}
            }
        ) for library in sentiment_libraries
    ]
    return ExperimentHandler(experiments)


def feature_comparison_handler():
    """ Compare using different combinations of sentiment features """
    features = [CompoundSentimentExperiment, SentimentExperiment,
                AggregatedCompoundSentimentExperiment, AggregatedSentimentExperiment,
                AggregatedCompoundPlusCountsSentimentExperiment, AggregatedPlusCountsSentimentExperiment,
                AggregatedCompoundPlusCountsPlusOverallSentimentExperiment,
                AggregatedPlusCountsPlusOverallSentimentExperiment]
    experiments = [
        (
            experiment,
            {
                "experiment_dir": "../training/statistical/sentiment/features",
                "experiment_name": experiment.__name__,
                "max_trials": 2,
                "num_cv_splits": 10,
                "hyperparameters": {"Sentiment.library": "vader", **default_svc_model}
            }
        ) for experiment in features
    ]
    return ExperimentHandler(experiments)


def model_hypertuning_handler():
    """ Using the best library and features, hyperparameter tune various models """
    experiments = [
        (
            AggregatedCompoundPlusCountsPlusOverallSentimentExperiment,
            {
                "experiment_dir": "../training/statistical/sentiment/hypertuning",
                "experiment_name": "AggregatedCompoundPlusCountsPlusOverallSentimentExperiment",
                "max_trials": 200,
                "num_cv_splits": 10,
                "hyperparameters": {"Sentiment.library": "vader"},
            }
        ), (
            AggregatedCompoundSentimentExperiment,
            {
                "experiment_dir": "../training/statistical/sentiment/hypertuning",
                "experiment_name": "AggregatedCompoundSentimentExperiment",
                "max_trials": 200,
                "num_cv_splits": 10,
                "hyperparameters": {"Sentiment.library": "vader"},
            }
        )
    ]
    return ExperimentHandler(experiments)


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    # Compare libraries
    library_handler = library_comparison_handler()
    # library_handler.run_experiments(dataset_dir)
    library_handler.print_results(2)

    # Compare features
    feature_handler = feature_comparison_handler()
    # feature_handler.run_experiments(dataset_dir)
    feature_handler.print_results(2)

    # Hyperparameter tuning
    hp_handler = model_hypertuning_handler()
    # hp_handler.run_experiments(dataset_dir)
    hp_handler.print_results(10)
