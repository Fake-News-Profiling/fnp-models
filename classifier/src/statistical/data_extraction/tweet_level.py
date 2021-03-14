import re

import nltk

import statistical.data_extraction.preprocessing as pre
from statistical.data_extraction.sentiment import analyzer

""" Tweet-level data extraction functions """


def tweet_level_extractor():
    """ Create a TweetStatsExtractor for tweet-level statistical features """
    extractor = pre.TweetStatsExtractor(extractors=[
        tag_counts,
        emojis_count,
        syllables_to_words_ratios,
        tweet_lengths,
        punctuation_counts,
        number_counts,
        personal_pronouns,
        char_to_words_ratio,
        quote_counts,
        capitalisation_counts,
        tweet_sentiment,
    ])
    extractor.feature_names = [
        # tag_counts
        "Number of '#USER#' tags",
        "Number of '#HASHTAG#' tags",
        "Number of '#URL#' tags",
        "Number of 'RT' tags",
        # emojis_count
        "Number of emojis",
        # syllables_to_words_ratios
        "Syllables-words ratio",
        # tweet_lengths
        "Tweet length in words",
        "Tweet length in characters",
        # punctuation_counts
        "Number of !",
        "Number of ?",
        "Number of ,",
        "Number of :",
        "Number of .",
        # number_counts
        "Number of numerical values",
        "Number of monetary values",
        # personal_pronouns
        "Number of personal pronouns",
        # char_to_words_ratio
        "Ratio of characters to words",
        # quote_counts
        "Number of quotes",
        # capitalisation_counts
        "Number of words with first letter capitalised",
        "Number of fully capitalised words",
        # tweet_sentiment
        "Compound sentiment"
    ]
    return extractor


def tag_counts(tweet, tags=None):
    """ Returns the number of tag used in this tweet, for each tag in tags """
    if tags is None:
        tags = ["#USER#", "#HASHTAG#", "#URL#", "RT"]
    return [tweet.count(tag) for tag in tags]


def emojis_count(tweet):
    """ Returns the number of emojis used in this tweet """
    return len(pre.emoji_chars([tweet])[0])


def syllables_to_words_ratios(tweet):
    """ Returns the number of syllables to the number of words in this tweet """
    tweet_words = pre.tweets_to_words([tweet], remove_tags=True)[0]
    tweet_syllables = sum(map(pre.syllables, tweet_words))
    return tweet_syllables / max(1, len(tweet_words))


def tweet_lengths(tweet):
    """ Return the length of the tweet, in words and characters """
    num_words = len(pre.tweets_to_words([tweet], remove_tags=True)[0])
    num_chars = len(pre.clean_text(tweet, remove_tags=True))
    return [num_words, num_chars]


def punctuation_counts(tweet, punctuation_marks="!?,:."):
    """
    Returns the number of each punctuation character in the tweet, for each punctuation character in punctuation_marks
    """
    return [tweet.count(punctuation) for punctuation in punctuation_marks]


def number_counts(tweet):
    """ Returns the number of numerical values and monetary values in the tweet """
    number_matcher = r"\d+(?:,\d+)*(?:\.\d+)?"
    numbers = len(re.findall(fr"(?:^| )(?<![£$€]){number_matcher}", tweet))
    money = len(re.findall(fr"[£$€]{number_matcher}", tweet))
    return [numbers, money]


def personal_pronouns(tweet):
    """ Returns the number of personal pronouns in the tweet """
    tweet_words = pre.tweets_to_words([tweet], remove_tags=True)[0]
    count = 0
    for tag in nltk.pos_tag(tweet_words):
        if tag[1] == 'PRP':
            count += 1

    return count


def char_to_words_ratio(tweet):
    """ Returns the ratio of characters to words in the tweet """
    clean_tweet = pre.clean_text(tweet, remove_digits=False, remove_tags=True)
    chars = len(clean_tweet)
    words = len(clean_tweet.split())
    chars -= words  # don't want to count spaces in chars
    return chars / max(1, words)


def quote_counts(tweet):
    """ Returns the number of quotes in the tweet """
    return len(re.findall("(?:^| )(?:“.*?”|‘.*?’|\".*?\"|\'.*?\')", tweet))


def capitalisation_counts(tweet):
    """ Returns the number of words with a capitalised first letter, or which are fully capitalised """
    clean_tweet = pre.clean_text(tweet, remove_digits=False, remove_tags=True)
    first_capitalised = len(re.findall(r"[A-Z][a-z]+", clean_tweet))
    fully_capitalised = len(re.findall(r"[A-Z]{2,}[^\w]", clean_tweet))
    return [first_capitalised, fully_capitalised]


def tweet_sentiment(tweet):
    """ Returns the compound sentiment of the tweet """
    return analyzer.polarity_scores(tweet)['compound']
