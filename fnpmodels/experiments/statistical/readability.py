import sys

from fnpmodels.processing.statistical.extractor import TweetStatsExtractor
from fnpmodels.processing.statistical import readability as read
from fnpmodels.experiments.experiment import ExperimentConfig
from fnpmodels.experiments.handler import ExperimentHandler
from fnpmodels.experiments.statistical import AbstractStatisticalExperiment, default_svc_model


""" Experiments for Readability models """


class AbstractReadabilityExperiment(AbstractStatisticalExperiment):
    """ Sklearn Experiment with 10 Cross-validation splits"""
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

    def input_data_transformer(self, x):
        return self.get_extractor().transform(x)


class StatisticalCountFeaturesExperiment(AbstractReadabilityExperiment):
    """
    Extract word/character statistical count features:
        * Tag counts
        * Emoji counts
        * Tweet lengths
        * Truncated tweets count
        * Numerical values count
    """

    @staticmethod
    def get_extractor():
        extractor = TweetStatsExtractor([
            read.tag_counts,
            lambda tweet_feed: read.emojis_counts(tweet_feed)[0],  # Return just the emoji counts
            read.average_tweet_lengths,
            read.truncated_tweets,
            read.number_counts,
        ])
        extractor.feature_names = [
            "Total number of '#USER#' tags",
            "Total number of '#HASHTAG#' tags",
            "Total number of '#URL#' tags",
            "Total number of emojis",
            "Average tweet lengths in words",
            "Average tweet lengths in characters",
            "Number of truncated tweets",
            "Total number of numerical values",
            "Total number of monetary values",
        ]
        return extractor


class TextReadabilityFeaturesExperiment(AbstractReadabilityExperiment):
    """
    Extract readability features:
        * Syllables-to-words ratio
        * Punctuation counts
        * Personal pronoun counts
        * Chars-to-words ratio
        * Word casing counts
    """

    @staticmethod
    def get_extractor():
        extractor = TweetStatsExtractor([
            read.syllables_to_words_ratios,
            lambda tweet_feed: read.punctuation_counts(tweet_feed)[:-1],  # Return just punctuation counts
            read.average_personal_pronouns,
            read.char_to_words_ratio,
            read.capitalisation_counts,
        ])
        extractor.feature_names = [
            "Total syllables-words ratio",
            "Mean syllables-words ratio",
            "Average number of !",
            "Average number of ?",
            "Average number of ,",
            "Average number of :",
            "Average number of .",
            "Average number of personal pronouns",
            "Total characters to words ratio",
            "Average words with first letter capitalised",
            "Average fully capitalised words",
        ]
        return extractor


class UniquenessFeaturesExperiment(AbstractReadabilityExperiment):
    """
    Extract uniqueness features:
        * Word ttr
        * Punctuation ttr
        * Emoji ttr
        * Retweets-to-tweets ratio
    """

    @staticmethod
    def get_extractor():
        extractor = TweetStatsExtractor([
            read.word_type_to_token_ratio,
            lambda tweet_feed: read.punctuation_counts(tweet_feed)[-1],  # Return just punctuation ttr
            lambda tweet_feed: read.emojis_counts(tweet_feed)[-1],  # Return just the emoji ttr
            read.retweet_ratio,
            read.quote_counts,
        ])
        extractor.feature_names = [
            "Total word type-token ratio",
            "Total punctuation type-token ratio",
            "Total emoji type-token ratio",
            "Ratio of retweets to tweets",
            "Total number of quotes",
        ]
        return extractor


class AllReadabilityFeaturesExperiment(AbstractReadabilityExperiment):
    """ Extract all readability features """

    @staticmethod
    def get_extractor():
        return read.readability_tweet_extractor()


def feature_comparison_handler():
    """ Compare using different combinations of readability features """
    features = [StatisticalCountFeaturesExperiment, TextReadabilityFeaturesExperiment,
                UniquenessFeaturesExperiment, AllReadabilityFeaturesExperiment]
    experiments = [
        (
            experiment,
            {
                "experiment_dir": "../training/statistical/readability/features",
                "experiment_name": experiment.__name__,
                "max_trials": 2,
                "num_cv_splits": 10,
                "hyperparameters": default_svc_model,
            }
        ) for experiment in features
    ]
    return ExperimentHandler(experiments)


def model_hypertuning_handler():
    """ Using the best features, hyperparameter tune various models """
    experiments = [
        (
            AllReadabilityFeaturesExperiment,
            {
                "experiment_dir": "../training/statistical/readability",
                "experiment_name": "hypertuning",
                "num_cv_splits": 10,
                "max_trials": 200,
            }
        )
    ]
    return ExperimentHandler(experiments)


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    # Compare features
    feature_handler = feature_comparison_handler()
    # library_handler.run_experiments(dataset_dir)
    feature_handler.print_results(2)

    # Hyperparameter tuning
    hp_handler = model_hypertuning_handler()
    # hp_handler.run_experiments(dataset_dir)
    hp_handler.print_results(5)
