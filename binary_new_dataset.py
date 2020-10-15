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

#shackleton-s => target
#dasovich-j => other

#change it with the appropriate path
dataset_path = "/home/giacomo/CNR/EmailAuthorship/enron_dataset/"
target_user = "dasovich-j"
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

#Shuffle dataset
c = list(zip(dataset, labels))
random.shuffle(c)
dataset_shuffled, labels_shuffled = zip(*c)

#Get 10% of testing set. If you want you can take 80% training, 10% validation, 10% testing
train = dataset_shuffled[0:math.floor(len(dataset_shuffled)*0.9)]
test = dataset_shuffled[math.floor(len(dataset_shuffled)*0.9)+1:]
y_train = labels_shuffled[0:math.floor(len(dataset_shuffled)*0.9)]
y_test = labels_shuffled[math.floor(len(dataset_shuffled)*0.9)+1:]

print("Training samples: " + str(len(y_train)))
print("Testing samples: " + str(len(y_test)))

#Tokenizer
tokenizer = Tokenizer(num_words=16000)
tokenizer.fit_on_texts(train)

X_train = tokenizer.texts_to_sequences(train)
#X_val = tokenizer.texts_to_sequences(val)
X_test = tokenizer.texts_to_sequences(test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


#pad
maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
#X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#Layers
embedding_dim = 100
'''
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen,
                           mask_zero=True))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(20, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
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


#Training
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

history = model.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_test, y_test), batch_size=64)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
ut.plot_history(history)
print(history.history)
