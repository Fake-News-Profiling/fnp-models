from fnpmodels.models import ScopedHyperParameters
from fnpmodels.models.ensemble import EnsembleBertModel
from fnpmodels.processing.parse import parse_dataset

if __name__ == "__main__":
    x, y = parse_dataset("../../datasets", "en")
    model = EnsembleBertModel(ScopedHyperParameters.from_json("ensemble_bert_hp.json"))
    # model.train(x, y)
