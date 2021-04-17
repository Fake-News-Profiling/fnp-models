import numpy as np

from fnpmodels.processing.preprocess import TweetPreprocessor, tag_indicators, replace_xml_and_html
from . import ScopedHyperParameters, AbstractModel
from .sklearn import SklearnModel
from .statistical import StatisticalModel
from .bert import BertModel


class EnsembleBertModel(AbstractModel):
    """ Combines multiple model predictions into one user classification """

    def __init__(self, hyperparameters: ScopedHyperParameters):
        super().__init__(hyperparameters)

        self.bert_model = BertModel(self.hp["bert"])
        self.stats_preprocessor = TweetPreprocessor([tag_indicators, replace_xml_and_html])
        self.readability_model = StatisticalModel(self.hp["readability"])
        self.sentiment_model = StatisticalModel(self.hp["sentiment"])
        self.ner_model = StatisticalModel(self.hp["ner"])
        self.ensemble_model = SklearnModel(self.hp["ensemble"])

    def _call_child_models(self, x):
        # BERT
        bert_out = self.bert_model(x)

        # Statistical
        stats_preprocessed = self.stats_preprocessor(x)
        read_out = self.readability_model(stats_preprocessed)
        sent_out = self.sentiment_model(stats_preprocessed)
        ner_out = self.ner_model(stats_preprocessed)
        return bert_out, read_out, sent_out, ner_out

    def __call__(self, x, *args, **kwargs):
        bert_out, read_out, sent_out, ner_out = self._call_child_models(x)
        x_outputs = np.concatenate([bert_out, read_out, sent_out, ner_out])
        return {
            "Ensemble.predict": self.ensemble_model.predict(x_outputs),
            "Ensemble.predict_proba": self.ensemble_model.predict_proba(x_outputs),
            "Ensemble.weights": self.ensemble_model.model.weights,  # TODO
            "Bert.predict_proba": bert_out,
            "Readability.predict_proba": read_out,
            "Sentiment.predict_proba": sent_out,
            "Ner.predict_proba": ner_out,
        }

    def train(self, x, y):
        # BERT
        self.bert_model.train(x, y)

        # Statistical
        xt = self.stats_preprocessor(x)
        self.readability_model.train(xt, y)
        self.sentiment_model.train(xt, y)
        self.ner_model.train(xt, y)

        # Fetch model data and train ensemble
        x_outputs = np.concatenate(self._call_child_models(x), axis=1)
        self.ensemble_model.train(x_outputs, y)







