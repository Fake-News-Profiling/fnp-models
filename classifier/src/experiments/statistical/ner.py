import sys
from functools import partial

from experiments.experiment import AbstractSklearnExperiment, ExperimentConfig
import statistical.data_extraction as ex
import statistical.data_extraction.ner.named_entity as ner
from experiments.handler import ExperimentHandler
from experiments.statistical import get_ner_wrapper
from statistical.data_extraction import TweetStatsExtractor

""" Experiments for NER models """


class NerCountsExperiment(AbstractSklearnExperiment):
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


class AggregatedNerCountsExperiment(AbstractSklearnExperiment):
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


if __name__ == "__main__":
    """ Execute experiments in this module """
    dataset_dir = sys.argv[1]

    experiments = [
        (
            NerCountsExperiment,
            {
                "experiment_dir": "../training/statistical/ner",
                "experiment_name": "counts_ner_spacy_sm",
                "max_trials": 100,
                "hyperparameters": {"Ner.library": "spacy", "Ner.spacy_pipeline": "en_core_web_sm"},
            }
        ), (
            AggregatedNerCountsExperiment,
            {
                "experiment_dir": "../training/statistical/ner",
                "experiment_name": "aggregated_ner_spacy_sm",
                "max_trials": 100,
                "hyperparameters": {"Ner.library": "spacy", "Ner.spacy_pipeline": "en_core_web_sm"},
            }
        )
    ]
    handler = ExperimentHandler(experiments)
    handler.run_experiments(dataset_dir)
    handler.print_results(10)
    handler.print_feature_importance(
        [
            NerCountsExperiment.get_extractor({"Ner.library": "spacy", "Ner.spacy_pipeline": "en_core_web_sm"}),
            AggregatedNerCountsExperiment.get_extractor({"Ner.library": "spacy", "Ner.spacy_pipeline": "en_core_web_sm"}),
        ],
        num_trials=10
    )
