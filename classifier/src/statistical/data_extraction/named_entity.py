from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Any

import spacy
from nltk.tag import StanfordNERTagger
from nltk.tokenize import TweetTokenizer

import statistical.data_extraction.preprocessing as pre


""" Named Entity Recognition model data extraction functions """


class AbstractNerTaggerWrapper(ABC):
    """ Named entity recognition wrapper to provide a uniform interface for NER tagger libraries """
    def __init__(self, tagger: Any, labels: List[str]):
        self.tagger = tagger
        self.labels = labels

    @abstractmethod
    def tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag some text, returning a list of tuples of tagged named entities, in the form:
        [(<entity_text>, <entity_label>) ...]
        """
        pass


class SpacyNerTaggerWrapper(AbstractNerTaggerWrapper):
    def __init__(self):
        tagger = spacy.load("en_core_web_sm")
        labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                  "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
        super().__init__(tagger, labels)

    def tag(self, text: str) -> List[Tuple[str, str]]:
        text_entities = self.tagger(text).ents
        return [(entity.text, entity.label_) for entity in text_entities]


class StanfordNerTaggerWrapper(AbstractNerTaggerWrapper):
    def __init__(self, model_filename: str, path_to_jar: str):
        tagger = StanfordNERTagger(model_filename, path_to_jar=path_to_jar)
        labels = ["PERSON", "ORGANIZATION", "LOCATION"]
        self.sl = set(labels)
        super().__init__(tagger, labels)
        self.tokenizer = TweetTokenizer()

    def tag(self, text: str) -> List[Tuple[str, str]]:
        words = self.tokenizer.tokenize(text)
        tags = self.tagger.tag(words)
        return tags


def ner_tweet_extractor(ner_library: str = "spacy", **kwargs):
    """ Create a TweetStatsExtractor for named entity recognition features """
    if ner_library == "spacy":
        # Use the Spacy NER tagger
        ner_tagger = SpacyNerTaggerWrapper()
    elif ner_library == "stanford":
        # Use the Stanford 3-class NER tagger
        ner_tagger = StanfordNerTaggerWrapper(**kwargs)
    else:
        raise ValueError("Invalid value for `ner_library`")

    extractor = pre.TweetStatsExtractor([partial(named_entities_counts, ner_tagger=ner_tagger)])
    extractor.feature_names = ner_tagger.labels
    return extractor


def named_entities_counts(tweet_feed: List[str], ner_tagger: AbstractNerTaggerWrapper = None) -> List[int]:
    """
    Extract the named entities from a users tweets, and return a list of counts for each entity
    """
    freq = dict.fromkeys(ner_tagger.labels, 0)

    for tweet in tweet_feed:
        cleaned_tweet = pre.clean_text(tweet, remove_punc=False, remove_digits=False, remove_tags=True)
        tags = ner_tagger.tag(cleaned_tweet)
        for _, label in tags:
            freq[label] += 1

    return list(freq.values())
