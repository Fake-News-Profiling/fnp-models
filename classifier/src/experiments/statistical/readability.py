import sys

from experiments.experiment import AbstractSklearnExperiment, ExperimentConfig
from statistical.data_extraction.preprocessing import TweetStatsExtractor
import statistical.data_extraction.readability as read
from experiments.handler import ExperimentHandler


""" Experiments for Readability models """


class ReadabilityExperiment(AbstractSklearnExperiment):
    """ Sklearn Experiment with 10 Cross-validation splits"""
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, num_cv_splits=10)

    def input_data_transformer(self, x):
        return self.get_extractor().transform(x)


class StatisticalCountFeaturesExperiment(ReadabilityExperiment):
    """
    Extract word/character statistical count features:
        * Tag counts
        * Emoji counts
        * Tweet lengths
        * Truncated tweets count
        * Numerical values count
    """

    def input_data_transformer(self, x):
        return self.get_extractor().transform(x)

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
            "Average number of '#USER#' tags per tweet",
            "Average number of '#HASHTAG#' tags per tweet",
            "Average number of '#URL#' tags per tweet",
            "Total number of emojis",
            "Average tweet lengths in words",
            "Average tweet lengths in characters",
            "Number of truncated tweets",
            "Total number of numerical values",
            "Total number of monetary values",
        ]
        return extractor


class TextReadabilityFeaturesExperiment(ReadabilityExperiment):
    """
    Extract readability features:
        * Syllables-to-words ratio
        * Punctuation counts
        * Personal pronoun counts
        * Chars-to-words ratio
        * Word casing counts
    """

    def input_data_transformer(self, x):
        return self.get_extractor().transform(x)

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
            "Ratio of characters to words",
            "Average words with first letter capitalised",
            "Average fully capitalised words",
        ]
        return extractor


class UniquenessFeaturesExperiment(ReadabilityExperiment):
    """
    Extract uniqueness features:
        * Word ttr
        * Punctuation ttr
        * Emoji ttr
        * Retweets-to-tweets ratio
    """

    def input_data_transformer(self, x):
        return self.get_extractor().transform(x)

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


class AllReadabilityFeaturesExperiment(ReadabilityExperiment):
    """ Extract all readability features """

    @staticmethod
    def get_extractor():
        return read.readability_tweet_extractor()


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            StatisticalCountFeaturesExperiment,
            {
                "experiment_dir": "../training/statistical/readability",
                "experiment_name": "statistical_counts",
                "max_trials": 100,
            }
        ), (
            TextReadabilityFeaturesExperiment,
            {
                "experiment_dir": "../training/statistical/readability",
                "experiment_name": "text_readability",
                "max_trials": 100,
            }
        ), (
            UniquenessFeaturesExperiment,
            {
                "experiment_dir": "../training/statistical/readability",
                "experiment_name": "uniqueness",
                "max_trials": 100,
            }
        ), (
            AllReadabilityFeaturesExperiment,
            {
                "experiment_dir": "../training/statistical/readability",
                "experiment_name": "all_features",
                "max_trials": 100,
            }
        )
    ]
    handler = ExperimentHandler(experiments)
    # handler.run_experiments(dataset_dir)
    handler.print_results(num_trials=10)
    handler.print_feature_importance(
        [
            StatisticalCountFeaturesExperiment.get_extractor(),
            TextReadabilityFeaturesExperiment.get_extractor(),
            UniquenessFeaturesExperiment.get_extractor(),
            AllReadabilityFeaturesExperiment.get_extractor(),
        ],
        num_trials=10)
