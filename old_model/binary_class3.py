import utils as ut
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import json
import random
import math
from time import time

dataset_path = "Dataset/json_dataset/"
target_user = "shackleton-s"
non_target_user = "dasovich-j"

def read_data():
	emails = []
	labels = []
	print(dataset_path + target_user + ".json")

	with open(dataset_path + target_user + ".json") as f:
		mails = json.load(f)
	for mail in mails["sent"]:
		emails.append(mail["body"])
		labels.append(1)
		
	with open(dataset_path + non_target_user + ".json") as f:
		mails = json.load(f)
	for mail in mails["sent"]:
		emails.append(mail["body"])
		labels.append(0)
	return emails, labels

#Read data from dataset
dataset, labels = read_data()
#print(len([elem for elem in labels if elem == 1]))

sys.exit()
#Shuffle dataset
c = list(zip(dataset, labels))
random.shuffle(c)
dataset_shuffled, labels_shuffled = zip(*c)

#Get 80% training, 10% validation, 10% testing
train = dataset_shuffled[0:math.floor(len(dataset_shuffled)*0.8)]
val = dataset_shuffled[math.floor(len(dataset_shuffled)*0.8)+1:math.floor(len(dataset_shuffled)*0.9)]
test = dataset_shuffled[math.floor(len(dataset_shuffled)*0.9)+1:]
y_train = labels_shuffled[0:math.floor(len(dataset_shuffled)*0.8)]
y_val = labels_shuffled[math.floor(len(dataset_shuffled)*0.8)+1:math.floor(len(dataset_shuffled)*0.9)]
y_test = labels_shuffled[math.floor(len(dataset_shuffled)*0.9)+1:]
print("Training samples: " + str(len(y_train)))
print("Val samples: " + str(len(y_val)))
print("Testing samples: " + str(len(y_test)))

#Tokenizer
tokenizer = Tokenizer(num_words=16000)
tokenizer.fit_on_texts(train)
tokenizer.fit_on_texts(val)
tokenizer.fit_on_texts(test)
#Save tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizerica.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
X_train = tokenizer.texts_to_sequences(train)
X_val = tokenizer.texts_to_sequences(val)
X_test = tokenizer.texts_to_sequences(test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
#pad
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#Layers
embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(vocab_size, 
                           embedding_dim, 
                           input_length=maxlen,     
                           trainable=True,
                           mask_zero=True))
model.add(layers.SpatialDropout1D(0.2))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

'''
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
keras.utils.plot_model(model, show_shapes=True)
'''

#Training
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

history = model.fit(X_train, y_train, 
                    epochs=10, 
                    verbose=1, 
                    validation_data=(X_val, y_val), 
                    batch_size=64)
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
model.save('Models/mod_3_erica_1')
ut.plot_history(history)
print(history.history)