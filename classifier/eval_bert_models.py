import sys

preprocessing_path = 'C:\\Users\\joshh\\Desktop\\Uni\\Soton Year 3\\COMP3200\\fake-news-profiling\\classifier\\preprocessing'
if preprocessing_path not in sys.path:
    sys.path.insert(1, preprocessing_path)

notif_path = 'C:\\Users\\joshh\\Desktop\\Uni\\Soton Year 3\\COMP3200\\fake-news-profiling\\classifier\\notifications'
if notif_path not in sys.path:
    sys.path.insert(1, notif_path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from official.nlp import optimization

import ipynb.fs.full.parse_datasets as datasets
import ipynb.fs.full.preprocessing as processing
import ipynb.fs.full.bert_fake_news_classifier as bclf
from ipynb.fs.full.notif_email import send_email

from data.data_parser import Tweet

# Load the saved dataset split
def load_data():
    return np.load("datasets/en_split_data-backup.npy", allow_pickle=True)

(tweet_train, label_train, 
 tweet_val, label_val, 
 tweet_test, label_test) = load_data()


# Preprocess dataset
print("Preprocessing data")
tweet_preprocessor = processing.BertTweetFeedDataPreprocessor(
    transformers = [
        processing.tag_indicators,
        processing.replace_xml_and_html,
        processing.replace_emojis,
        processing.remove_punctuation,
        processing.replace_tags,
        processing.remove_hashtag_chars,
        processing.replace_accented_chars,
        processing.tag_numbers,
        processing.remove_stopwords,
        processing.remove_extra_spacing,
    ])
tweet_train_processed = tweet_preprocessor.transform(tweet_train)
tweet_val_processed = tweet_preprocessor.transform(tweet_val)
tweet_test_processed = tweet_preprocessor.transform(tweet_test)


bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"
bert_size = 128
batch_size = 32
epochs = 16
optimizer_name = "adamw"
lr = 5e-5

print("Making model eval")
model_handler = bclf.BertModelEvalHandler(
    bert_url, bert_size, bclf.BertTweetFeedTokenizer, bclf.dense_bert_model)

print("Training")
train_history = model_handler.train_bert(
    tweet_train_processed,
    label_train,
    batch_size,
    epochs,
    tweet_val_processed,
    label_val,
    optimizer_name,
    lr,
    "training/bert_clf/initial_eval/bert_ffnn_1/test/cp.ckpt",
    "training/bert_clf/initial_eval/bert_ffnn_1/test",
)


# BERT Feed Model with 256 input size
# print("\n\n\nBERT Feed Model with 256 input size")
# bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"
# bert_size = 256
# models = [(batch_size, epochs, lr, optimizer_name)
#          for batch_size in [32]
#          for epochs in [10]
#          for lr in [3e-5, 2e-5]
#          for optimizer_name in ['adam', 'adamw']]
#
# model_path = "training/bert_feed/initial_eval/"
#
#
# for batch_size, epochs, lr, optimizer_name in models:
#     with tf.device(('/gpu:0' if batch_size < 16 else '/cpu:0')):
#         model_name = f"bert256-batch_size{batch_size}-epochs{epochs}-lr{lr}-optimizer{optimizer_name}"
#
#
#         text = f"Finished training {model_name}, training history was:\n{train_history.history}"
#         print(text)
#         send_email(text)
#         with open(model_path+"train_history.txt", "a") as file:
#             def join_hist_array(array):
#                 return ",".join([str(i) for i in array])
#
#             name = f"F_{bert_size}_{batch_size}_{str(lr)[0]}_{'A' if optimizer_name == 'adam' else 'AW'}"
#             file_text = ", ,".join([
#                 ",".join([str(bert_size), str(batch_size), str(lr), optimizer_name, name]),
#                 join_hist_array(train_history.history['loss']),
#                 join_hist_array(train_history.history['val_loss']),
#                 join_hist_array(train_history.history['binary_accuracy']),
#                 join_hist_array(train_history.history['val_binary_accuracy']),
#             ]) + "\n"
#             file.write(file_text)
#
#
# # BERT Individual Model with 256 input size
# print("\n\n\nBERT Individual Model with 256 input size")
# bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"
# bert_size = 256
# models = [(batch_size, epochs, lr, optimizer_name)
#          for batch_size in [8, 16, 24, 32]
#          for epochs in [10]
#          for lr in [5e-5, 3e-5, 2e-5]
#          for optimizer_name in ['adam', 'adamw']]
#
# model_path = "training/bert_individual/initial_eval/"
#
# for batch_size, epochs, lr, optimizer_name in models:
#     with tf.device(('/gpu:0' if batch_size < 16 else '/cpu:0')):
#         model_name = f"bert256-batch_size{batch_size}-epochs{epochs}-lr{lr}-optimizer{optimizer_name}"
#         model_handler = bclf.BertModelEvalHandler(
#             bert_url, bert_size, bclf.BertIndividualTweetTokenizer, bclf.dense_bert_model)
#
#         train_history = model_handler.train_bert(
#             tweet_train_processed,
#             label_train,
#             batch_size,
#             epochs,
#             tweet_val_processed,
#             label_val,
#             optimizer_name,
#             lr,
#             model_path + model_name + "/cp.ckpt",
#             model_path + "logs/" + model_name,
#         )
#
#         text = f"Finished training {model_name}, training history was:\n{train_history.history}"
#         print(text)
#         send_email(text)
#         with open(model_path+"train_history.txt", "a") as file:
#             def join_hist_array(array):
#                 return ",".join([str(i) for i in array])
#
#             name = f"I_{bert_size}_{batch_size}_{str(lr)[0]}_{'A' if optimizer_name == 'adam' else 'AW'}"
#             file_text = ", ,".join([
#                 ",".join([str(bert_size), str(batch_size), str(lr), optimizer_name, name]),
#                 join_hist_array(train_history.history['loss']),
#                 join_hist_array(train_history.history['val_loss']),
#                 join_hist_array(train_history.history['binary_accuracy']),
#                 join_hist_array(train_history.history['val_binary_accuracy']),
#             ]) + "\n"
#             file.write(file_text)
