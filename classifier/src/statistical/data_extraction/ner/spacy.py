from typing import List, Tuple

import spacy

from statistical.data_extraction import AbstractNerTaggerWrapper


class SpacyNerTaggerWrapper(AbstractNerTaggerWrapper):
    def __init__(self, spacy_pipeline: str):
        tagger = spacy.load(spacy_pipeline)
        labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                  "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
        super().__init__(tagger, labels)

    def tag(self, text: str) -> List[Tuple[str, str]]:
        text_entities = self.tagger(text).ents
        return [(entity.text, entity.label_) for entity in text_entities]