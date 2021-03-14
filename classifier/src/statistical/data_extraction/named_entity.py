import spacy

import statistical.data_extraction.preprocessing as pre

spacy_nlp = spacy.load("en_core_web_sm")
spacy_ner_labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                    "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]


""" Named Entity Recognition model data extraction functions """


def ner_tweet_extractor():
    """ Create a TweetStatsExtractor for named entity recognition features """
    extractor = pre.TweetStatsExtractor(extractors=[named_entities_counts])
    extractor.feature_names = spacy_ner_labels
    return extractor


def named_entities_counts(user_tweets):
    """
    Extract the named entities from a users tweets, and return an array of counts for each entity
    """
    freq = dict.fromkeys(spacy_ner_labels, 0)
    norp_counts = []

    for i, tweet in enumerate(user_tweets):
        cleaned_tweet = pre.clean_text(tweet, remove_punc=False, remove_digits=False, remove_tags=True)
        tweet_ne = spacy_nlp(cleaned_tweet).ents
        norp_count = 0
        for entity in tweet_ne:
            freq[entity.label_] += 1
            if entity.label_ == "NORP":
                norp_count += 1

        norp_counts.append(norp_count)

    return list(freq.values())
