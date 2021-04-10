from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Any

import numpy as np

import statistical.data_extraction.preprocessing as pre


""" Named Entity Recognition model data extraction functions """


class AbstractNerTaggerWrapper(ABC):
    """ Named entity recognition wrapper to provide a uniform interface to NER tagger libraries """

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


def ner_tweet_extractor(ner_wrapper: AbstractNerTaggerWrapper) -> pre.TweetStatsExtractor:
    """ Create a TweetStatsExtractor for named entity recognition features """
    extractor = pre.TweetStatsExtractor([partial(named_entities_counts, ner_tagger=ner_wrapper)])
    extractor.feature_names = ner_wrapper.labels
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


def aggregated_named_entities_counts(tweet_feed: List[str], ner_tagger: AbstractNerTaggerWrapper = None) -> List[int]:
    """
    Extract the named entities from a users tweets, and for each entity return it's:
    total count, average count per tweet, count range, standard deviation per tweet
    """
    label_indices = {label: i for i, label in enumerate(ner_tagger.labels)}
    counts = np.full((len(tweet_feed), len(label_indices)), 0)

    for i, tweet in enumerate(tweet_feed):
        cleaned_tweet = pre.clean_text(tweet, remove_punc=False, remove_digits=False, remove_tags=True)
        tags = ner_tagger.tag(cleaned_tweet)
        for _, label in tags:
            counts[i, label_indices[label]] += 1

    results = []
    for i in range(counts.shape[-1]):
        count = np.sum(counts[:, i])
        mean = np.mean(counts[:, i])
        tweet_range = np.max(counts[:, i]) - np.min(counts[:, i])
        std = np.std(counts[:, i])
        results += [count, mean, tweet_range, std]

    return results
