import sys
from functools import partial

from experiments.experiment import ExperimentConfig
import statistical.data_extraction.ner.named_entity as ner
from experiments.handler import ExperimentHandler
from experiments.statistical import get_ner_wrapper, AbstractStatisticalExperiment, default_svc_model, sklearn_models
from statistical.data_extraction import TweetStatsExtractor

""" Experiments for NER models """


class NerCountsExperiment(AbstractStatisticalExperiment):
    """ Extract named-entity recognition data at the user-level, and use this to train an Sklearn model """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, num_cv_splits=10)

    def input_data_transformer(self, x):
        return self.get_extractor(self.hyperparameters).transform(x)

    @staticmethod
    def get_extractor(hp):
        ner_wrapper = get_ner_wrapper(hp)
        extractor = TweetStatsExtractor([partial(ner.named_entities_counts, ner_tagger=ner_wrapper)])
        extractor.feature_names = ner_wrapper.labels
        return extractor


class AggregatedNerCountsExperiment(AbstractStatisticalExperiment):
    """ Extract named-entity recognition data at the user-level, and use this to train an Sklearn model """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, num_cv_splits=10)

    def input_data_transformer(self, x):
        return self.get_extractor(self.hyperparameters).transform(x)

    @staticmethod
    def get_extractor(hp):
        ner_wrapper = get_ner_wrapper(hp)
        extractor = TweetStatsExtractor([partial(ner.aggregated_named_entities_counts, ner_tagger=ner_wrapper)])
        extractor.feature_names = [f"{label}_{s}" for label in ner_wrapper.labels
                                   for s in ["count", "mean", "range", "std"]]
        return extractor


def library_comparison_handler():
    """ Comparing VADER, Stanza and TextBlob libraries """
    sentiment_libraries = ["nltk", "stanza"]
    experiments = [
        (
            NerCountsExperiment,
            {
                "experiment_dir": "../training/statistical/ner/library_comparison",
                "experiment_name": library,
                "max_trials": 2,
                "hyperparameters": {"Ner.library": library, **default_svc_model},
            }
        ) for library in sentiment_libraries
    ]
    return ExperimentHandler(experiments)


def feature_comparison_handler():
    """ Compare using different combinations of sentiment features """
    features = [NerCountsExperiment, AggregatedNerCountsExperiment]
    experiments = [
        (
            experiment,
            {
                "experiment_dir": "../training/statistical/ner/features",
                "experiment_name": experiment.__name__,
                "max_trials": 2,
                "hyperparameters": {"Ner.library": "stanza", **default_svc_model}
            }
        ) for experiment in features
    ]
    return ExperimentHandler(experiments)


def model_hypertuning_handler():
    """ Using the best library and features, hyperparameter tune various models """
    experiments = [
        (
            NerCountsExperiment,
            {
                "experiment_dir": "../training/statistical/ner/hypertuning",
                "experiment_name": model,
                "max_trials": 200,
                "hyperparameters": {"Ner.library": "stanza", "Sklearn.model_type": model},
            }
        ) for model in sklearn_models
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
    hp_handler.run_experiments(dataset_dir)
    hp_handler.print_results(10)
