from collections import Counter
from functools import partial
import re

import numpy as np
import nltk

import statistical.data_extraction.preprocessing as pre


""" Readability model data extraction functions """


def readability_tweet_extractor():
    """ Create a TweetStatsExtractor for readability features """
    extractor = pre.TweetStatsExtractor(extractors=[
        tag_counts,
        retweet_ratio,
        emojis_count,
        syllables_to_words_ratios,
        average_tweet_lengths,
        word_type_to_token_ratio,
        truncated_tweets,
        punctuation_counts,
        number_counts,
        average_personal_pronouns,
        char_to_words_ratio,
        quote_counts,
        capitalisation_counts,
    ])
    extractor.feature_names = [
        # tag_counts
        "Average number of '#USER#' tags per tweet",
        "Average number of '#HASHTAG#' tags per tweet",
        "Average number of '#URL#' tags per tweet",
        # retweet_ratio
        "Ratio of retweets to tweets",
        # emojis_count
        "Total number of emojis",
        "Total emoji type-token ratio",
        # syllables_to_words_ratios
        "Total syllables-words ratio",
        "Mean syllables-words ratio",
        "Min syllables-words ratio",
        "Max syllables-words ratio",
        # average_tweet_lengths
        "Average tweet lengths in words",
        "Average tweet lengths in characters",
        # word_type_to_token_ratio
        "Total word type-token ratio",
        # truncated_tweets
        "Number of truncated tweets",
        # punctuation_counts
        "Average number of !",
        "Average number of ?",
        "Average number of ,",
        "Average number of :",
        "Total punctuation type-token ratio",
        # number_counts
        "Total number of numerical values",
        "Total number of monetary values",
        # average_personal_pronouns
        "Average number of personal pronouns",
        # char_to_words_ratio
        "Ratio of characters to words",
        # quote_counts
        "Total number of quotes",
        # capitalisation_counts
        "Average words with first letter capitalised",
        "Average fully capitalised words",
    ]
    return extractor


def tag_counts(user_tweets, tags=None):
    """ Returns the average number of tag used, for each tag in tags """
    if tags is None:
        tags = ['#USER#', '#HASHTAG#', '#URL#']
    return np.mean([[tweet.count(tag) for tag in tags] for tweet in user_tweets], axis=0)


def retweet_ratio(user_tweets):
    """ Returns the ratio of retweets to regular tweets """
    retweets = 0
    second_retweets = 0
    for tweet in user_tweets:
        if tweet.startswith("RT"):
            retweets += 1

    return retweets / len(user_tweets)


def emojis_count(user_tweets):
    """ Returns the following emoji counts for this user: total number of emojis used, average number of emojis used
    per tweet, type-token ratio of emojis (uniqueness of emojis used) """
    tweet_emojis = pre.emoji_chars(user_tweets)
    flattened_tweet_emojis = pre.flatten(tweet_emojis)

    total_num_emojis = len(flattened_tweet_emojis)
    emoji_type_token_ratio = (len(Counter(flattened_tweet_emojis)) / total_num_emojis) if total_num_emojis > 0 else 0
    return np.asarray([total_num_emojis, emoji_type_token_ratio])


def syllables_to_words_ratios(user_tweets):
    """ Returns the overall, average, min, and max ratios of the number of syllables to the number of words """
    tweet_words = pre.tweets_to_words(user_tweets, remove_tags=True)
    tweet_syllables = [sum(map(pre.syllables, words)) for words in tweet_words]
    per_tweet_ratios = [tweet_syllables[i] / max(1, len(tweet_words[i])) for i in range(len(tweet_words))]

    overall_ratio = sum(tweet_syllables) / max(1, sum(map(len, tweet_words)))
    mean_ratio = np.mean(per_tweet_ratios)
    min_ratio = min(per_tweet_ratios)
    max_ratio = max(per_tweet_ratios)
    return np.asarray([overall_ratio, mean_ratio, min_ratio, max_ratio])


def average_tweet_lengths(user_tweets):
    """ Returns the average tweet lengths in words and characters """
    mean_words = np.mean(list(map(len, pre.tweets_to_words(user_tweets, remove_tags=True))))
    mean_chars = np.mean(list(map(len, map(partial(pre.clean_text, remove_tags=True), user_tweets))))
    return np.asarray([mean_words, mean_chars])


def word_type_to_token_ratio(user_tweets):
    """ Returns the ratio of unique words to the total number of words in all of a users tweets """
    words = pre.flatten(pre.tweets_to_words(user_tweets, remove_tags=True))
    return len(Counter(list(words))) / len(words)


def truncated_tweets(user_tweets):
    """ Returns the number of truncated tweets """
    count = 0
    for tweet in user_tweets:
        if re.match(r".*\.\.\.(?: #URL#)?$", tweet) is not None:
            count += 1

    return count


def punctuation_counts(user_tweets, punctuation_marks="!?,:"):
    """ Returns the average number of each punctuation character in the users tweets, for each punctuation character
    in punctuation_marks. Also returns the punctuation type-to-token ratio of all of the users tweets """
    all_punc = [c for tweet in user_tweets
                for c in pre.clean_text(tweet, remove_punc=False, remove_tags=True) if c in pre.punctuation]
    punc_ttr = len(Counter(all_punc)) / max(1, len(all_punc))
    punc_counts = [[tweet.count(punctuation) for punctuation in punctuation_marks] for tweet in user_tweets]
    mean_punc_counts = np.mean(punc_counts, axis=0)
    return np.concatenate([mean_punc_counts, [punc_ttr]])


def number_counts(user_tweets):
    """ Returns the following counts: number of numerical values in the users tweets (e.g. "7,000"), number of
    monetary values in the users tweets (e.g. "$90,000", "£9.35") """
    number_matcher = r"\d+(?:,\d+)*(?:\.\d+)?"
    numbers = sum([len(re.findall(fr"(?:^| )(?<![£$€]){number_matcher}", tweet)) for tweet in user_tweets])
    money = sum([len(re.findall(fr"[£$€]{number_matcher}", tweet)) for tweet in user_tweets])
    return np.asarray([numbers, money])


def average_personal_pronouns(user_tweets):
    """ Returns the average number of personal pronouns per tweets """
    personal_pronouns_count = []
    for tweet_words in pre.tweets_to_words(user_tweets, remove_tags=True):
        count = 0
        for tag in nltk.pos_tag(tweet_words):
            if tag[1] == 'PRP':
                count += 1

        personal_pronouns_count.append(count)

    return np.mean(personal_pronouns_count)


def char_to_words_ratio(user_tweets):
    """ Returns the ratio of characters to words in the users tweets """
    chars = 0
    words = 0
    for tweet in user_tweets:
        cleaned_tweet = pre.clean_text(tweet, remove_digits=False, remove_tags=True)
        chars += len(cleaned_tweet)
        words += len(cleaned_tweet.split())

    chars -= words  # don't want to count spaces in chars
    return chars / max(1, words)


def quote_counts(user_tweets):
    """ Returns the total and average number of quotes used by the user """
    num_quotes = [len(re.findall("(?:^| )(?:“.*?”|‘.*?’|\".*?\"|\'.*?\')", tweet)) for tweet in user_tweets]
    return sum(num_quotes)


def capitalisation_counts(user_tweets):
    """ Returns the following counts: average number of words with a capitalised first letter,
    average number of fully capitalised words """
    first_capitalised = []
    fully_capitalised = []
    for tweet in user_tweets:
        cleaned_tweet = pre.clean_text(tweet, remove_tags=True)
        first_capitalised.append(len(re.findall(r"[A-Z][a-z]+", cleaned_tweet)))
        fully_capitalised.append(len(re.findall(r"[A-Z]{2,}[^\w]", cleaned_tweet)))

    return np.asarray([
        np.mean(first_capitalised),
        np.mean(fully_capitalised),
    ])
