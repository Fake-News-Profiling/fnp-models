from fnpmodels.models import ScopedHyperParameters
from fnpmodels.models.ensemble import EnsembleBertModel
from fnpmodels.processing.parse import parse_dataset

if __name__ == "__main__":
    x, y = parse_dataset("../datasets", "en")
    model = EnsembleBertModel(ScopedHyperParameters.from_json("fnpmodels/ensemble_bert_hp.json"))
    print(model([x[0]]), y[0])
    # model.train(x, y)
