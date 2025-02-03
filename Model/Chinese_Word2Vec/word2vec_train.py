import os
import jieba
import re
import pickle
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


RESULT_FILE_PATH = './data_proc/res.pkl'
MODEL_SAVE_PATH = './model/embedding1.model'

with open('./data_proc/res.pkl' ,'rb') as file:
    train_data = pickle.load(file)
model = Word2Vec(train_data,
                 sg=0,
                 vector_size=128,
                 min_count=1,
                 epochs=10,
                 window=5)
model.train(train_data ,epochs=model.epochs ,total_examples=model.corpus_count)
model.wv.save_word2vec_format(MODEL_SAVE_PATH ,binary=True)
