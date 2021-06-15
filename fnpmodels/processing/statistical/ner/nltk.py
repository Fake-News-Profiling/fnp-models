from typing import List, Tuple
from operator import itemgetter

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tag import StanfordNERTagger

from . import AbstractNerTaggerWrapper


class NltkNerTaggerWrapper(AbstractNerTaggerWrapper):
    def __init__(self):
        labels = ["PERSON", "ORGANIZATION", "LOCATION", "GPE", "FACILITY", "GSP"]
        super().__init__(None, labels)

    def tag(self, text: str) -> List[Tuple[str, str]]:
        tokens = word_tokenize(text)
        pos = pos_tag(tokens)
        return list(
            map(
                lambda chunk: (" ".join(map(itemgetter(0), chunk)), chunk.label()),
                filter(lambda chunk: hasattr(chunk, "label"), ne_chunk(pos))
            )
        )


class NltkStanfordNerTaggerWrapper(AbstractNerTaggerWrapper):
    def __init__(self, classifier_path: str, jar_path: str):
        tagger = StanfordNERTagger(classifier_path, jar_path)
        labels = ["PERSON", "ORGANIZATION", "LOCATION"]
        super().__init__(tagger, labels)
        self.counter = 0

    def tag(self, text: str) -> List[Tuple[str, str]]:
        self.counter += 1
        print(self.counter)
        if self.counter % 1000 == 0:
            print("%d done" % self.counter)

        tokens = word_tokenize(text)
        return list(filter(lambda ent: ent[1] != "O", self.tagger.tag(tokens)))
