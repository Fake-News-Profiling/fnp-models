from abc import abstractmethod

import tensorflow as tf
from official.nlp.bert.tokenization import FullTokenizer
from tensorflow_hub import KerasLayer

from fnpmodels.models import AbstractProcessor


class AbstractBertTokenizer(AbstractProcessor):
    """ Abstract BERT Tokenizer """
    label_pattern = None

    def __init__(self, encoder, bert_hidden_layer_size):
        """ Create the BERT encoder and tokenizer """
        self.bert_hidden_layer_size = bert_hidden_layer_size
        self.tokenizer = FullTokenizer(
            encoder.resolved_object.vocab_file.asset_path.numpy(),
            do_lower_case=encoder.resolved_object.do_lower_case.numpy()
        )

    def __call__(self, *args, **kwargs):
        return self.tokenize_input(*args, **kwargs)

    @abstractmethod
    def tokenize_input(self, x, **kwargs):
        """ Tokenize input data """
        pass

    def tokenize_labels(self, y, user_label_pattern=True):
        """ Tokenize input data labels """
        if self.label_pattern is not None and user_label_pattern:
            labels = [int(v) for v, n in zip(y, self.label_pattern) for _ in range(n)]
            return tf.convert_to_tensor(labels, tf.int32)
        elif not user_label_pattern:
            return tf.convert_to_tensor([int(v) for v in y], tf.int32)
        else:
            raise Exception("blah blah")

    def _format_bert_tokens(self, ragged_word_ids):
        """ Create, format and pad BERT's input tensors """
        # Generate mask, and pad word_ids and mask
        mask = tf.ones_like(ragged_word_ids).to_tensor()
        word_ids = ragged_word_ids.to_tensor()
        padding = tf.constant([[0, 0], [0, (self.bert_hidden_layer_size - mask.shape[1])]])
        word_ids = tf.pad(word_ids, padding, "CONSTANT")
        mask = tf.pad(mask, padding, "CONSTANT")
        type_ids = tf.zeros_like(mask)

        return {
            'input_word_ids': word_ids,
            'input_mask': mask,
            'input_type_ids': type_ids,
        }

    def _format_bert_word_piece_input(self, word_piece_tokens):
        word_piece_tokens.insert(0, '[CLS]')
        word_piece_tokens.append('[SEP]')
        return self.tokenizer.convert_tokens_to_ids(word_piece_tokens)


class BertIndividualTweetTokenizer(AbstractBertTokenizer):
    """ BERT tokenizer which tokenizes historical tweet data as individual tweets """

    def tokenize_input(self, x, **kwargs):
        """ Tokenize input data """
        tokenized_tweets = [
            self._tokenize_single_tweet(tweet) for tweet_feed in x for tweet in tweet_feed
        ]
        self.label_pattern = [len(tweet_feed) for tweet_feed in x]
        word_ids = tf.ragged.constant(tokenized_tweets)
        return self._format_bert_tokens(word_ids)

    def _tokenize_single_tweet(self, tweet):
        """ Tokenize a single tweet, truncating its tokens to bert_input_size """
        tokens = self.tokenizer.tokenize(tweet)[:self.bert_hidden_layer_size - 2]
        return self._format_bert_word_piece_input(tokens)


class BertTweetFeedTokenizer(AbstractBertTokenizer):
    """ BERT tokenizer which tokenizes historical tweet data as tweet feed chunks """

    def tokenize_input(self, x, overlap=50):
        """ Tokenize input data """
        tokenized_tweet_feeds = [
            self._tokenize_tweet_feed(" ".join(tweet_feed), overlap) for tweet_feed in x
        ]
        self.label_pattern = [len(tweet_feed) for tweet_feed in tokenized_tweet_feeds]
        flattened_feeds = [chunk for feed in tokenized_tweet_feeds for chunk in feed]
        word_ids = tf.ragged.constant(flattened_feeds)
        return self._format_bert_tokens(word_ids)

    def _tokenize_tweet_feed(self, tweet_feed, overlap):
        """ Tokenize an entire tweet feed into chunks """
        feed_tokens = self.tokenizer.tokenize(tweet_feed)
        tokens = [
            feed_tokens[i:i + self.bert_hidden_layer_size - 2]
            for i in range(0, len(feed_tokens), self.bert_hidden_layer_size - overlap)
        ]

        return list(map(self._format_bert_word_piece_input, tokens))

    @staticmethod
    def get_data_len(X_data_lens, bert_size, overlap=50):
        return sum([
            (sum(tweet_feed) + len(tweet_feed) - 1) / (bert_size - 2 - overlap) for tweet_feed in X_data_lens
        ])


def bert_tokenizer(encoder_url, hidden_layer_size, tokenizer_class):
    encoder = KerasLayer(encoder_url, trainable=False)
    return tokenizer_class(encoder, hidden_layer_size)


def tokenize_bert_input(encoder_url, hidden_layer_size, tokenizer_class, x_train, y_train, x_val=None, y_val=None,
                        shuffle=False, feed_overlap=50, return_tokenizer=False):
    tokenizer = bert_tokenizer(encoder_url, hidden_layer_size, tokenizer_class)
    x_train = tokenizer.tokenize_input(x_train, overlap=feed_overlap)
    y_train = tokenizer.tokenize_labels(y_train)

    if shuffle:
        x_train, y_train = shuffle_bert_data(x_train, y_train)

    result = [x_train, y_train]

    if x_val is not None:
        x_val = tokenizer.tokenize_input(x_val, overlap=feed_overlap)
        y_val = tokenizer.tokenize_labels(y_val) if y_val is not None else None
        result += [x_val, y_val]

    if return_tokenizer:
        result.append(tokenizer)

    return result


def shuffle_bert_data(x, y, seed=1):
    y_shuffled = _shuffle_tensor_with_seed(y, seed=seed)
    x_shuffled = {k: _shuffle_tensor_with_seed(v, seed=seed) for k, v in x.items()}
    return x_shuffled, y_shuffled


def _shuffle_tensor_with_seed(tensor, seed=1):
    tf.random.set_seed(1)
    return tf.random.shuffle(tensor, seed=seed)


def tokenize_x(hp, tokenizer, x, shuffle=True):
    def tokenize_tweet_feed(tweet_feed):
        data = [list(tokenizer.tokenize_input([[tweet]]).values()) for tweet in tweet_feed]
        return tf.random.shuffle(data) if shuffle else data

    x_tok = tf.convert_to_tensor(list(map(tokenize_tweet_feed, x)))
    x_chunked = tf.reshape(x_tok, shape=(-1, hp.get("Bert.tweet_feed_len"), 3, hp.get("Bert.hidden_size")))
    # shape(-1, 100, 3, 128) => shape(-1, TWEET_FEED_LEN, 3, 128)
    return x_chunked


def tokenize_y(hp, y):
    return tf.convert_to_tensor([v for v in y for _ in range(100 // hp.get("Bert.tweet_feed_len"))])
