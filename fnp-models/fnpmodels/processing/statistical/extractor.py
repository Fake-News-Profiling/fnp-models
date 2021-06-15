from collections.abc import Iterable
import string
import re
from typing import List, Callable, Union

import numpy as np
from sklearn.preprocessing import normalize
from pyphen import Pyphen
import demoji

from fnpmodels.models import AbstractProcessor

""" A collection of statistical extraction and pre-processing functions """


digits = set("0123456789")
printable = set(string.printable)
punctuation = set(string.punctuation)
punctuation.remove('#')
pyphen = Pyphen(lang='en')


def clean_text(text: str,
               remove_punc: bool = True,
               remove_non_print: bool = True,
               remove_emojis: bool = True,
               remove_digits: bool = True,
               remove_tags: bool = False):
    """ Clean text by removing certain characters (e.g. punctuation) """
    if remove_emojis:
        text = demoji.replace(text, "")

    chars = []
    for char in text:
        if not ((remove_punc and char in punctuation) or
                (remove_non_print and char not in printable) or
                (remove_digits and char in digits)):
            chars.append(char)

    cleaned = "".join(chars)
    if remove_tags:
        return re.sub('#[A-Z]+#', "", cleaned)

    return cleaned


def tweets_to_words(tweet_feed: List[str], **kwargs) -> List[List[str]]:
    """ Cleans each tweet, and splits by spaces to turn them into lists of words """
    return [clean_text(tweet, **kwargs).split() for tweet in tweet_feed]


def emoji_chars(tweet_feed: List[str]) -> List[List[str]]:
    """ Returns a list of lists of emojis used in each of the users tweets """
    return [demoji.findall_list(tweet) for tweet in tweet_feed]


def syllables(word: str) -> int:
    """ Counts the number of syllables in a word """
    return pyphen.inserted(word).count('-') + 1


def flatten(list_of_lists: List[List]) -> List:
    """ Flatten a 2-dimensional list """
    return [x for list_ in list_of_lists for x in list_]


class TweetStatsExtractor(AbstractProcessor):
    """
    Given a list of data extracting functions, this will apply these functions to inputted data, returning a feature
    vector
    """
    feature_names = []

    def __init__(self, extractors: List[Callable]):
        if len(extractors) == 0:
            raise Exception("Must pass at least one extracting function")

        self.extractors = extractors

    def __call__(self, x: Union[List[List[str]], List[str], np.ndarray],
                 normalize_data: bool = False,
                 *args, **kwargs) -> np.ndarray:
        result = []
        for user_tweets in x:
            if len(self.extractors) > 1:
                result.append(np.concatenate([self._apply_extractor(f, user_tweets) for f in self.extractors]))
            else:
                result.append(self._apply_extractor(self.extractors[0], user_tweets))

        return np.asarray(normalize(result) if normalize_data else result)

    @staticmethod
    def _apply_extractor(extractor: Callable, data: Union[List[List[str]], List[str], np.ndarray]) -> np.ndarray:
        result = extractor(data)
        if isinstance(result, Iterable):
            return np.asarray(result)
        else:
            return np.asarray([result])
