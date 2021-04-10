from typing import List, Tuple

import stanza

from statistical.data_extraction import AbstractNerTaggerWrapper


class StanzaNerTaggerWrapper(AbstractNerTaggerWrapper):
    def __init__(self):
        tagger = stanza.Pipeline(lang="en", processors="tokenize,ner")
        labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                  "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
        super().__init__(tagger, labels)
        self.counter = 0

    def tag(self, text: str) -> List[Tuple[str, str]]:
        self.counter += 1
        if self.counter % 1000 == 0:
            print("%d done" % self.counter)

        text_entities = self.tagger(text).ents
        return [(entity.text, entity.type) for entity in text_entities]
