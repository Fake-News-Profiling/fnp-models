import sys

from experiments.experiment import AbstractSklearnExperiment
import statistical.data_extraction as ex
from experiments.handler import ExperimentHandler

""" Experiments for individual statistical models """


class ReadabilityExperiment(AbstractSklearnExperiment):
    """ Extract readability data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        extractor = ex.readability_tweet_extractor()
        return extractor.transform(x)


class NerExperiment(AbstractSklearnExperiment):
    """ Extract named-entity recognition data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        ner_library = self.hyperparameters.get("Ner.library")
        model_filename = path_to_jar = None
        if ner_library == "stanford":
            model_filename = self.hyperparameters.get("Ner.model_filename")
            path_to_jar = self.hyperparameters.get("Ner.path_to_jar")
        extractor = ex.ner_tweet_extractor(ner_library, model_filename=model_filename, path_to_jar=path_to_jar)
        return extractor.transform(x)


class SentimentExperiment(AbstractSklearnExperiment):
    """ Extract sentiment data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        extractor = ex.sentiment_tweet_extractor()
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
        #         "experiment_name": "ner_spacy_7",
        #         "max_trials": 100,
        #         "hyperparameters": {"Ner.library": "spacy"},
        #     }
        # ), (
            NerExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "ner_stanford_7",
                "max_trials": 100,
                "hyperparameters": {
                    "Ner.library": "stanford",
                    "Ner.model_filename": "../../../stanford_ner/english.all.3class.distsim.crf.ser.gz",
                    "Ner.path_to_jar": "../../../stanford_ner/stanford-ner-4.2.0.jar"
                },
            }
        ), (
            SentimentExperiment,
            {
                "experiment_dir": "../training/statistical",
                "experiment_name": "sentiment_7",
                "max_trials": 100,
            }
        )
    ]
    handler = ExperimentHandler(experiments)
    handler.run_experiments(dataset_dir)
    handler.print_results()
