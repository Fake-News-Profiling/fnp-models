import sys

from experiments.experiment import AbstractSklearnExperiment
import statistical.data_extraction as ex
from experiments.handler import ExperimentHandler
from experiments.statistical import get_ner_wrapper, get_sentiment_wrapper

""" Experiments for individual statistical models """


class ReadabilityExperiment(AbstractSklearnExperiment):
    """ Extract readability data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        extractor = ex.readability_tweet_extractor()
        return extractor.transform(x)


class NerExperiment(AbstractSklearnExperiment):
    """ Extract named-entity recognition data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        ner_wrapper = get_ner_wrapper(self.hyperparameters)
        extractor = ex.ner_tweet_extractor(ner_wrapper)
        return extractor.transform(x)


class SentimentExperiment(AbstractSklearnExperiment):
    """ Extract sentiment data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)
        extractor = ex.sentiment_tweet_extractor(sentiment_wrapper)
        return extractor.transform(x)


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
        #     ReadabilityExperiment,
        #     {
        #         "experiment_dir": "../training/statistical",
        #         "experiment_name": "readability_7",
        #         "max_trials": 100,
        #     }
        # ), (
        #     NerExperiment,
        #     {
        #         "experiment_dir": "../training/statistical",
        #         "experiment_name": "ner_spacy_sm_7",
        #         "max_trials": 100,
        #         "hyperparameters": {"Ner.library": "spacy", "Ner.spacy_pipeline": "en_core_web_sm"},
        #     }
        # ), (
            SentimentExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "sentiment_vader_7",
                "max_trials": 100,
                "hyperparameters": {"Sentiment.library": "vader"}
            }
        ), (
            SentimentExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "sentiment_stanza_7",
                "max_trials": 100,
                "hyperparameters": {"Sentiment.library": "stanza"}
            }
        ), (
            SentimentExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "sentiment_textblob_7",
                "max_trials": 100,
                "hyperparameters": {"Sentiment.library": "textblob"}
            }
        )
    ]
    handler = ExperimentHandler(experiments)
    handler.run_experiments(dataset_dir)
    handler.print_results()
