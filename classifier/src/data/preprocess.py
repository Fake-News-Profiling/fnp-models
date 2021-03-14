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
    hashtags_tagged = re.sub(r"#[a-zA-Z\d]+\s", "#HASHTAG#", tweet, flags=re.MULTILINE)
    urls_tagged = re.sub(
        r"https?://[a-zA-Z.\d_]*", "#URL#", hashtags_tagged, flags=re.MULTILINE
    )
    users_tagged = re.sub(r"@[a-zA-Z\d_]+", "#USER#", urls_tagged, flags=re.MULTILINE)
    return users_tagged


def replace_xml_and_html(tweet):
    """ Replace XML encodings (&amp; &lt; &gt;) and HTML tags (<br>) """
    replace_xml = unescape(tweet)
    replace_html = BeautifulSoup(replace_xml, features="lxml").get_text()
    replace_html = replace_html.replace("\xa0", " ")  # Remove non-breaking spaces '\xa0'
    return replace_html


def remove_punctuation_and_non_printables(tweet):
    """ Remove punctuation, except hashtags '#' """
    punctuation = re.sub(r"[#:]", "", string.punctuation)
    printable = re.sub(fr"[{punctuation}]", "", string.printable)
    remove_quotes = tweet.replace("'", "")  # Replace quotes with an empty string, so "I've" -> "Ive
    return re.sub(fr"[^{printable}]", " ", remove_quotes)


def remove_colons(tweet):
    """ Remove colons """
    return tweet.replace(":", " ")


def replace_emojis(tweet, with_desc=True, sep=":"):
    """ Replace emojis with their descriptions ':smiling face:', or with an emoji tag '[emoji]' """
    if with_desc:
        return demoji.replace_with_desc(tweet, sep)
    else:
        return demoji.replace(tweet, "[emoji]")


def remove_emojis(tweet):
    """ Remove emojis """
    return demoji.replace(tweet, " ")


def replace_tags(tweet, remove=False):
    """ Replace #HASHTAG# and #URL# #USER# with tags [tag] and [url], [user] """
    tweet = tweet.replace(r"#HASHTAG#", " " if remove else "[hashtag]")
    tweet = tweet.replace(r"#URL#", " " if remove else "[url]")
    tweet = tweet.replace(r"#USER#", " " if remove else "[user]")
    # Repeated sub for 'RT' to handle multiple continuous retweets
    tweet = re.sub(r"((?:^|\s)RT\s)", " " if remove else " [retweet] ", tweet)
    tweet = re.sub(r"((?:^|\s)RT\s)", " " if remove else " [retweet] ", tweet)
    return tweet


def remove_hashtags(tweet):
    """ Remove hashtags '#' """
    return "".join(c for c in tweet if c != "#")


def replace_unicode(tweet):
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
    return re.sub(r"\s+", " ", tweet).strip()


class BertTweetPreprocessor:
    """ Pre-processes tweet feeds to be used in a BERT-based tweet embedding model """

    def __init__(self, transformers=None):
        if transformers is None:
            transformers = [
                tag_indicators,
                replace_xml_and_html,
                remove_colons,  # Remove colons as they're used as emoji separators
                replace_emojis,  # Replace emojis with ':<emoji_description>:'
                replace_unicode,
                remove_punctuation_and_non_printables,  # Remove punctuation except : and #
                replace_tags,
                remove_hashtags,
                tag_numbers,
                remove_extra_spacing,
            ]
        self.transformers = transformers

    def transform(self, X):
        """ Preprocess a list of user tweets """
        return np.asarray([
            np.asarray([self._transform_single_tweet(tweet) for tweet in tweet_feed])
            for tweet_feed in X
        ])

    def _transform_single_tweet(self, tweet):
        """ Preprocess a single tweet """
        return reduce(lambda data, transformer: transformer(data), self.transformers, tweet)
