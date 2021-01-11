import re
import string
from xml.sax.saxutils import unescape

import demoji
import numpy as np
from bs4 import BeautifulSoup
from unidecode import unidecode


class TweetPreprocessor:
    def __init__(self, tweets):
        """ Constructs a new InputPreprocessor object, given a list of tweets. """
        self.tweets = list(map(lambda tweet: tweet.text, tweets))
        self._preprocess()

    def _preprocess(self):
        # tag urls, hashtags, user mentions
        def tag_indicators(tweet):
            hashtags_tagged = re.sub(r"#[^\s]*", "#HASHTAG#", tweet, flags=re.MULTILINE)
            urls_tagged = re.sub(
                r"https?\:\/\/[^\s]*", "#URL#", hashtags_tagged, flags=re.MULTILINE
            )
            users_tagged = re.sub(r"@[^\s]*", "#USER#", urls_tagged, flags=re.MULTILINE)
            return users_tagged

        # Replace XML encodings - &amp; &lt; &gt;
        # Then remove HTML tags - <br>
        def replace_xml_and_html(tweet):
            replace_xml = unescape(tweet)
            replace_html = BeautifulSoup(replace_xml, features="lxml").get_text()
            return replace_html

        # Remove punctuation, apart from '#'
        def remove_punctuation(tweet):
            punc = set(string.punctuation)
            prin = set(string.printable)
            punc.remove("#")
            return "".join(c for c in tweet if c not in punc and c in prin)

        # Replace emojis with their meaning, e.g. :smiling_face:
        def replace_emojis(tweet):
            return demoji.replace_with_desc(tweet, ":")

        # Replace #HASHTAG# and #URL# #USER# with tags [tag] and [url], [user]
        def replace_tags(tweet):
            tweet = tweet.replace("#HASHTAG#", "[tag]")
            tweet = tweet.replace("#URL#", "[url]")
            tweet = tweet.replace("#USER#", "[user]")
            return tweet

        # Remove '#'s
        def remove_hashtag_chars(tweet):
            return "".join(c for c in tweet if c != "#")

        def replace_accented_chars(tweet):
            return unidecode(tweet)

        def remove_extra_spacing(tweet):
            return " ".join(tweet.split())

        preprocess_funcs = [
            tag_indicators,
            replace_xml_and_html,
            replace_emojis,
            remove_punctuation,
            replace_tags,
            remove_hashtag_chars,
            replace_accented_chars,
            remove_extra_spacing,
        ]

        # Process an individual tweet
        def process_tweet(tweet):
            for f in preprocess_funcs:
                tweet = f(tweet)

            return tweet

        self.tweets_processed = [process_tweet(tweet) for tweet in self.tweets]

    def get_individual_tweets_dataset(self):
        """ Returns an array of preprocessed tweets. """
        return np.asarray(self.tweets_processed)

    def get_tweet_feed_dataset(self):
        """ Concatenated all tweets, returning them as one string in an array. """
        return np.asarray([" ".join(self.tweets_processed)])
