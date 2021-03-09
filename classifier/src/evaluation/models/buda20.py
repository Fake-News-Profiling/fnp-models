import re
from statistics import pstdev

import numpy as np
from emoji import EMOJI_UNICODE_ENGLISH
from lexical_diversity import lex_div as ld
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from evaluation.models import AbstractEvaluationModel

"""
Academic Integrity Statement:
All of the code in this file has been adapted from (Buda & Bolonyai, 2020)'s submission to the PAN 2020 Fake News 
Author Profiling task. This has been done purely for evaluation purposes, to compare my work to theirs.
* The original code: https://github.com/pan-webis-de/buda20
* Buda & Bolonyai, 2020, paper: https://pan.webis.de/downloads/publications/papers/buda_2020.pdf
"""


class Buda20NgramEnsembleModel(AbstractEvaluationModel):
    def __init__(self):
        # Ngram Models
        self.lr_ngram = Pipeline([
            ('vect', TfidfVectorizer(min_df=6, ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('lr', LogisticRegression(C=1000, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0))
        ])
        self.rf_ngram = Pipeline([
            ('vect', TfidfVectorizer(min_df=9, ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=9, criterion='gini'))
        ])
        self.svm_ngram = Pipeline([
            ('vect', TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True, use_idf=True, smooth_idf=True)),
            ('rf', SVC(C=100, kernel='linear', verbose=False, probability=True))
        ])
        self.xgb_ngram = Pipeline([
            ('vect', TfidfVectorizer(min_df=8, ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('xgb', XGBClassifier(colsample_bytree=0.6, eta=0.01, max_depth=6, n_estimators=300, subsample=0.8))
        ])

        # Descriptive Statistical Model
        self.xgb_stats = XGBClassifier(
            colsample_bynode=1,
            colsample_bytree=0.9,
            gamma=2,
            learning_rate=0.2,
            max_depth=2,
            min_child_weight=4,
            n_estimators=200,
            reg_alpha=0.1,
            subsample=0.8,
        )

        # Final LogisticRegression ensemble classifier
        self.lr_ensemble = LogisticRegression(
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5,
        )

    def fit(self, x_train, y_train):
        x_train_clean_v1, x_train_clean_v2, x_train_stats = self._clean_data(x_train)

        # Train models
        self.lr_ngram.fit(x_train_clean_v1, y_train)
        self.rf_ngram.fit(x_train_clean_v2, y_train)
        self.svm_ngram.fit(x_train_clean_v1, y_train)
        self.xgb_ngram.fit(x_train_clean_v1, y_train)
        self.xgb_stats.fit(x_train_stats, y_train)

        # Train final model
        model_preds = self._model_predictions(x_train_clean_v1, x_train_clean_v2, x_train_stats)
        self.lr_ensemble.fit(model_preds, y_train)

    def predict(self, x):
        model_preds = self._model_predictions(*self._clean_data(x))
        return self.lr_ensemble.predict(model_preds)

    def _model_predictions(self, x_train_clean_v1, x_train_clean_v2, x_train_stats):
        # Predict probabilities, for final model
        pred_lr_ngram = self.lr_ngram.predict_proba(x_train_clean_v1)[:, 1]
        pred_rf_ngram = self.rf_ngram.predict_proba(x_train_clean_v2)[:, 1]
        pred_svm_ngram = self.svm_ngram.predict_proba(x_train_clean_v1)[:, 1]
        pred_xgb_ngram = self.xgb_ngram.predict_proba(x_train_clean_v1)[:, 1]
        pred_xgb_stats = self.xgb_stats.predict_proba(x_train_stats)[:, 1]
        return np.concatenate([pred_lr_ngram, pred_rf_ngram, pred_svm_ngram, pred_xgb_ngram, pred_xgb_stats])

    @staticmethod
    def _clean_data(x_train):
        x_train_clean_v1 = cleaning_v1(x_train)
        x_train_clean_v2 = cleaning_v2(x_train)
        x_train_stats = extract_features(flatten_tweet_feeds(x_train))
        return x_train_clean_v1, x_train_clean_v2, x_train_stats


# Data cleaning
def cleaning_v1(tweet_lists):
    cleaned_feed_v1 = []
    for feed in tweet_lists:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        cleaned_feed_v1.append(feed)
    return cleaned_feed_v1


def cleaning_v2(tweet_lists):
    cleaned_feed_v2 = []
    for feed in tweet_lists:
        feed = feed.lower()
        feed = ''.join(' ' + char if char in EMOJI_UNICODE_ENGLISH else char for char in feed).strip()
        feed = re.sub('[,.\'\"\‘\’\”\“]', '', feed)
        feed = re.sub(r'([a-z\'-\’]+)', r'\1 ', feed)
        feed = re.sub(r'(?<![?!:;/])([:\'\";.,?()/!])(?= )', '', feed)
        feed = re.sub('[\n]', ' ', feed)
        feed = ' '.join(feed.split())
        cleaned_feed_v2.append(feed)
    return cleaned_feed_v2


# Feature engineering (methods expect flattened tweet feeds)
def flatten_tweet_feeds(tweet_feeds):
    return np.asarray([tweet for user_feed in tweet_feeds for tweet in user_feed])


def extract_features(tweet_feeds):
    # Tweet lengths
    len_tw_char = [len(i) for i in tweet_feeds]
    len_tw_word = [len(i.split(" ")) for i in tweet_feeds]

    # SD
    len_char_sd_auth = [pstdev(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
    len_word_sd_auth = [pstdev(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]

    # min - max - range - mean
    len_char_min_auth = [min(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
    len_word_min_auth = [min(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]

    len_char_max_auth = [max(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
    len_word_max_auth = [max(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]

    len_char_rng_auth = [max(len_tw_char[i * 100:i * 100 + 99]) - min(len_tw_char[i * 100:i * 100 + 99]) for
                         i in range(int(len(len_tw_char) / 100))]
    len_word_rng_auth = [max(len_tw_word[i * 100:i * 100 + 99]) - min(len_tw_word[i * 100:i * 100 + 99]) for
                         i in range(int(len(len_tw_word) / 100))]

    len_char_mean_auth = [np.mean(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
    len_word_mean_auth = [np.mean(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]

    # Vocab variety (TTR)
    tweets_szerz = [" ".join(list(tweet_feeds)[i * 100:99 + i * 100]) for i in range(int(len(len_tw_char) / 100))]
    ttr_szerz = [ld.ttr(ld.flemmatize(i)) for i in tweets_szerz]

    # Tags (RT, URL, Hashtag, User, truncated, emojis)
    rt_szerz = [np.sum([k == "RT" for k in i.split(" ")]) for i in tweets_szerz]
    url_szerz = [np.sum([k == "#URL#" for k in i.split(" ")]) for i in tweets_szerz]
    hsg_szerz = [np.sum([k == "#HASHTAG#" for k in i.split(" ")]) for i in tweets_szerz]
    user_szerz = [np.sum([k == "#USER#" for k in i.split(" ")]) for i in tweets_szerz]
    p_szerz = [np.sum([k[-1:] == "…" for k in i.split(" ")]) for i in tweets_szerz]

    emoj_szerz = []
    for aut in tweets_szerz:
        emdb = 0
        for tok in aut.split(" "):
            for c in tok:
                emdb += c in EMOJI_UNICODE_ENGLISH
        emoj_szerz.append(emdb)

    return np.asarray([
        len_char_sd_auth,
        len_word_sd_auth,
        len_char_min_auth,
        len_word_min_auth,
        len_char_max_auth,
        len_word_max_auth,
        len_char_rng_auth,
        len_word_rng_auth,
        len_char_mean_auth,
        len_word_mean_auth,
        rt_szerz,
        url_szerz,
        hsg_szerz,
        user_szerz,
        p_szerz,
        emoj_szerz,
        ttr_szerz,
    ])
