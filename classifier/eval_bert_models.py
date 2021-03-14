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

# Preprocess search
def preprocess_funcs_search(processing_funcs, model_params, X_train, y_train, X_val, y_val, save_path):
    for i, funcs in enumerate(processing_funcs):
        preprocessor = processing.BertTweetFeedDataPreprocessor(transformers=funcs)
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        with open(save_path + "train_history.txt", "a") as file:
            file.write(f"\n-----Preprocessing funcs are: {funcs}")

        for params in model_params:
            if len(funcs) == 0 and params["batch_size"] != 80:
                continue
            bert_size = params["bert_size"]
            lr = params["learning_rate"]
            optimizer_name = params["optimizer"]
            batch_size = params["batch_size"]
            epochs = 10
            name = f"{i}-{params['model_type']}_{bert_size}_{batch_size}_{str(lr)[0]}_{'A' if optimizer_name == 'adam' else 'AW'}"
            print("\n", name, "\n")
            
            with tf.device(('/gpu:0' if batch_size < 32 else '/cpu:0')):
                model_handler = bclf.BertModelEvalHandler(
                    params["bert_url"], bert_size, params["tokenizer"], bclf.dense_bert_model)

                train_history = model_handler.train_bert(
                    X_train_processed,
                    y_train,
                    batch_size,
                    epochs,
                    X_val_processed,
                    y_val,
                    optimizer_name,
                    lr,
                    save_path + name + "/cp.ckpt",
                    save_path + "logs/" + name,
                )

            def join_hist_array(array):
                return ",".join([str(i) for i in array])

            file_text = ", ,".join([
                ",".join([str(bert_size), str(batch_size), str(lr), optimizer_name, name]), 
                join_hist_array(train_history.history['loss']),
                join_hist_array(train_history.history['val_loss']),
                join_hist_array(train_history.history['binary_accuracy']),
                join_hist_array(train_history.history['val_binary_accuracy']),
            ])
            with open(save_path + "train_history.txt", "a") as file:
                file.write("\n"+file_text)
    
        send_email(f"Finished training for preprocessing funcs:\n{funcs}")
        

preprocess_funcs_search(
    processing_funcs=[
        # No preprocessing functions (raw data)
        [],
        # Remove HTML, replace emojis, remove accented chars, replace tags
        [processing.replace_xml_and_html,
         processing.replace_emojis, 
         processing.replace_accented_chars, 
         processing.replace_tags, 
         processing.remove_extra_spacing],
        # Remove HTML, replace emojis, remove accented chars, replace tags, remove punctuation, tag numbers
        [processing.replace_xml_and_html,
         processing.replace_emojis, 
         processing.replace_accented_chars, 
         processing.remove_punctuation,
         processing.replace_tags,
         processing.remove_hashtag_chars,
         processing.tag_numbers,
         processing.remove_extra_spacing],
        # Remove HTML, replace emojis, remove accented chars, replace tags, remove punctuation, tag numbers, remove stopwords
        [processing.replace_xml_and_html,
         processing.replace_emojis, 
         processing.replace_accented_chars, 
         processing.remove_punctuation,
         processing.replace_tags,
         processing.remove_hashtag_chars,
         processing.tag_numbers,
         processing.remove_stopwords,
         processing.remove_extra_spacing],
    ],
    model_params=[
        # F_128_8_2_AW
        {"model_type": "F",
         "bert_size": 128, 
         "learning_rate": 2e-5, 
         "optimizer": "adamw", 
         "batch_size": 8,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"},
        # F_128_16_2_AW
        {"model_type": "F",
         "bert_size": 128, 
         "learning_rate": 2e-5, 
         "optimizer": "adamw", 
         "batch_size": 16,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"},
        # F_128_24_2_AW
        {"model_type": "F",
         "bert_size": 128, 
         "learning_rate": 2e-5, 
         "optimizer": "adamw", 
         "batch_size": 24,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"},
        # F_128_32_3_AW
        {"model_type": "F",
         "bert_size": 128, 
         "learning_rate": 3e-5, 
         "optimizer": "adamw", 
         "batch_size": 32,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"},
        # F_128_32_5_AW
        {"model_type": "F",
         "bert_size": 128, 
         "learning_rate": 5e-5, 
         "optimizer": "adamw", 
         "batch_size": 32,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"},
        # F_128_40_5_AW
        {"model_type": "F",
         "bert_size": 128, 
         "learning_rate": 5e-5, 
         "optimizer": "adamw", 
         "batch_size": 40,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1"},
        # F_256_24_5_AW
        {"model_type": "F",
         "bert_size": 256, 
         "learning_rate": 5e-5, 
         "optimizer": "adamw", 
         "batch_size": 24,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"},
        # F_256_64_5_AW
        {"model_type": "F",
         "bert_size": 256, 
         "learning_rate": 5e-5, 
         "optimizer": "adamw", 
         "batch_size": 64,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"},
        # F_256_80_5_AW
        {"model_type": "F",
         "bert_size": 256, 
         "learning_rate": 5e-5, 
         "optimizer": "adamw", 
         "batch_size": 80,
         "tokenizer": bclf.BertTweetFeedTokenizer, 
         "bert_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"},
    ],
    X_train=tweet_train,
    y_train=label_train,
    X_val=tweet_val,
    y_val=label_val,
    save_path="training/preprocessing/initial_eval/",
)

send_email("Finished training")
