import json, os, math
import csv
import utils
import tensorflow as tf
from tensorboard.plugins import projector
import sys
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import random
from time import time
import datetime
import matplotlib.pyplot as plt
from keras import layers 
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def plot(indexes, heats, words):
	plt.figure(figsize=(12, 5))
	plt.bar(indexes, heats)
	for i in indexes:
		plt.text(i-.25, max(0, heats[i]*1.05), words[i], rotation=90)
	plt.title('Erica')
	plt.show()

def print_samples():
    pred = model.predict_classes(X_test, verbose = 2)
    for i in range(len(y_test)-1) :
        print('Expected:', y_train[i], 'Predicted', pred[i]) 

def concat(x):
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    concatenated = layers.Concatenate(axis=1)([avg_pool, max_pool])
    return concatenated

def read_data(base_dir, target_user):
	t_emails = []
	t_labels = []
	nt_labels = []
	nt_emails = []
	print(base_dir + target_user + ".json")

	with open(base_dir + target_user + ".json") as f:
		mails = json.load(f)
	for mail in mails["sent"]:
		if(mail["body"] == ""):
			continue
		to_clean = mail["body"]
		clean = to_clean.replace('\r', ' \r ')
		t_emails.append(clean)
		t_labels.append(1)

	for filename in os.listdir(base_dir):
		with open(base_dir + filename) as f:
			mails = json.load(f)
		for mail in mails["sent"]:
			if((filename == (target_user + ".json")) or mail["body"] == ""):
				continue
			to_clean = mail["body"]
			clean = to_clean.replace('\r', ' \r ')
			nt_emails.append(clean)
			nt_labels.append(0)
	return t_emails, t_labels, nt_emails, nt_labels

def shuffle_data(dataset, labels):
	c = list(zip(dataset, labels))
	random.shuffle(c)
	return zip(*c)

def save_tokenizer_in(path, tokenizer):
	tokenizer_json = tokenizer.to_json()
	with open(path, 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def split_dataset(dataset_shuffled, trainPerc, valPerc, testPerc):
	train = dataset_shuffled[0:math.floor(len(dataset_shuffled)*trainPerc)]
	val = dataset_shuffled[math.floor(len(dataset_shuffled)*trainPerc)+1:math.floor(len(dataset_shuffled)*(1-testPerc))]
	test = dataset_shuffled[math.floor(len(dataset_shuffled)*(1-testPerc))+1:]
	return train, val, test

