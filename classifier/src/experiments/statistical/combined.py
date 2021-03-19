from experiments.experiment import AbstractSklearnExperiment
import statistical.data_extraction as ex
from experiments.statistical import get_sentiment_wrapper, get_ner_wrapper

""" Experiment for training combined and ensemble statistical models """


class CombinedStatisticalExperiment(AbstractSklearnExperiment):
    """ Extract all statistical data at the user-level, and use this to train an Sklearn model """

    def input_data_transformer(self, x):
        ner_wrapper = get_ner_wrapper(self.hyperparameters)
        sentiment_wrapper = get_sentiment_wrapper(self.hyperparameters)
        extractor = ex.combined_tweet_extractor(ner_wrapper=ner_wrapper, sentiment_wrapper=sentiment_wrapper)
        return extractor.transform(x)


class EnsembleStatisticalExperiment(AbstractSklearnExperiment):
    """ Load and train statistical models, and then train an Sklearn ensemble model """

    def input_data_transformer(self, x):
        pass
