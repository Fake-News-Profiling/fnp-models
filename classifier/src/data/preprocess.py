from functools import reduce
import re
import string

import numpy as np
from xml.sax.saxutils import unescape
import demoji
from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.corpus import stopwords


def tag_indicators(tweet):
    """ Replace URLs, hastags and user mentions with a tag (e.g. #HASHTAG#) """
    hashtags_tagged = re.sub(r"#[^\s]*", "#HASHTAG#", tweet, flags=re.MULTILINE)
    urls_tagged = re.sub(
        r"https?\:\/\/[^\s]*", "#URL#", hashtags_tagged, flags=re.MULTILINE
    )
    users_tagged = re.sub(r"@[^\s]*", "#USER#", urls_tagged, flags=re.MULTILINE)
    return users_tagged


def replace_xml_and_html(tweet):
    """ Replace XML encodings (&amp; &lt; &gt;) and HTML tags (<br>) """
    replace_xml = unescape(tweet)
    replace_html = BeautifulSoup(replace_xml, features="lxml").get_text()
    return replace_html


def remove_punctuation(tweet):
    """ Remove punctuation, except hashtags '#' """
    punc = set(string.punctuation)
    prin = set(string.printable)
    punc.remove("#")
    return "".join(c for c in tweet if c not in punc and c in prin)


def replace_emojis(tweet):
    """ Replace emojis with their meaning ':smiling_face:' """
    return demoji.replace_with_desc(tweet, ":")


def replace_tags(tweet):
    """ Replace #HASHTAG# and #URL# #USER# with tags [tag] and [url], [user] """
    tweet = tweet.replace("#HASHTAG#", "[hashtag]")
    tweet = tweet.replace("#URL#", "[url]")
    tweet = tweet.replace("#USER#", "[user]")
    tweet = tweet.replace("RT", "[retweet]")
    return tweet


def remove_hashtag_chars(tweet):
    """ Remove hashtags '#' """
    return "".join(c for c in tweet if c != "#")


def replace_accented_chars(tweet):
    """ Replace accented characters with their ASCII equivalent """
    return unidecode(tweet)


def tag_numbers(tweet):
    """ Replace sequences of digits with 'number' """
    return re.sub("[0-9]+", "number", tweet)


def remove_stopwords(tweet):
    """ Remove english stopwords """
    tweet_words = list(
        filter(
            lambda word: word.lower() not in stopwords.words('english'), tweet.split(" ")
        )
    )
    return " ".join(tweet_words)


def remove_extra_spacing(tweet):
    """ Remove extra spaces """
    return " ".join(tweet.split())


class BertTweetFeedDataPreprocessor:
    """ Preprocesses tweet feeds to be used in the BertIndividualTweetModel """

    def __init__(self, transformers=None):
        if transformers is None:
            transformers = [
                tag_indicators,
                replace_xml_and_html,
                replace_emojis,
                remove_punctuation,
                replace_tags,
                remove_hashtag_chars,
                replace_accented_chars,
                tag_numbers,
                remove_stopwords,
                remove_extra_spacing,
            ]
        self.transformers = transformers

    def transform(self, X):
        """ Preprocess a list of user tweets """
        return np.asarray([
            np.asarray([self._transform_single_tweet(tweet.text) for tweet in tweet_feed])
            for tweet_feed in X
        ])

    def _transform_single_tweet(self, tweet):
        """ Preprocess a single tweet """
        return reduce(lambda data, transformer: transformer(data), self.transformers, tweet)
