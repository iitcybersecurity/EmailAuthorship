import json, os, math
import csv
from utils import *
import tensorflow as tf
from tensorboard.plugins import projector
import sys
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import random
from time import time
import datetime

def prova_1():
    model = Sequential()
    model.add(layers.Embedding(vocab,
                            embedding_dim, 
                            input_length=maxlen, 
                            weights=[matrix],
                            trainable=train,
                            mask_zero=True,
                            input_shape=(50,)))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_embedding_model(vocab, embedding_dim, maxlen, matrix = None, train = True):
    model = Sequential()
    model.add(layers.Embedding(vocab,
                            embedding_dim,
                            input_length=maxlen,
                            weights=[matrix],
                            trainable=train,
                            mask_zero=True,
                            ))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def embedding_model(vocab, embedding_dim, maxlen, matrix = None, train = True):
    model = Sequential()
    model.add(layers.Embedding(vocab, 
                            embedding_dim, 
                            input_length=maxlen,  
                            weights=[matrix],    
                            trainable=train,
                            mask_zero=True))
    model.add(layers.SpatialDropout1D(0.2))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def create_embedding_matrix(filepath, word_index, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

