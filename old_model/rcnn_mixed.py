import json, os, math
import csv
import utils as ut
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

target_user = "shackleton-s"
non_target_user = "dasovich-j"
base_dir = "Dataset/RAW/enron_dataset/"
log_dir = "Models/logs/erica_mixed/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def read_data():
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

def save_tokenizer_in(path):
	tokenizer_json = tokenizer.to_json()
	with open(path, 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def split_dataset(dataset_shuffled, trainPerc, valPerc, testPerc):
	train = dataset_shuffled[0:math.floor(len(dataset_shuffled)*trainPerc)]
	val = dataset_shuffled[math.floor(len(dataset_shuffled)*trainPerc)+1:math.floor(len(dataset_shuffled)*(1-testPerc))]
	test = dataset_shuffled[math.floor(len(dataset_shuffled)*(1-testPerc))+1:]
	return train, val, test


#Read data from dataset
t_dataset, t_labels, nt_dataset, nt_labels = read_data()
t_len = len([elem for elem in t_labels])
nt_len = len([elem for elem in nt_labels])

#Shuffle dataset
nt_dataset_shuffled, nt_labels_shuffled = shuffle_data(nt_dataset, nt_labels)
dataset = t_dataset + nt_dataset[0:t_len]
labels = t_labels + nt_labels[0:t_len]
dataset_shuffled, labels_shuffled = shuffle_data(dataset, labels)

#Get 80% training, 10% validation, 10% testing
train, val, test = split_dataset(dataset_shuffled, 0.8, 0.1, 0.1)
y_train, y_val, y_test = split_dataset(labels_shuffled, 0.8, 0.1, 0.1)

print("Training samples: " + str(len(y_train)))
print("Val samples: " + str(len(y_val)))
print("Testing samples: " + str(len(y_test)))

#Tokenizer
tokenizer = Tokenizer(num_words=16000)
tokenizer.fit_on_texts(train)
tokenizer.fit_on_texts(val)
tokenizer.fit_on_texts(test)

#salve tokenizer
#save_tokenizer_in('tokenizerica_mixed.json')

X_train = tokenizer.texts_to_sequences(train)
X_val = tokenizer.texts_to_sequences(val)
X_test = tokenizer.texts_to_sequences(test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
word_index = tokenizer.word_index
subwords = word_index

with open('word_index.txt', 'w') as f:
   print(word_index, file=f)

#logs metadata
# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
	f.write("Word\tLabel\n")
	for key, value in subwords.items():
		f.write("{}\t{}\n".format(str(key), value))
  # Fill in the rest of the labels with "unknown"
  for unknown in range(1, vocab_size - len(subwords)):
    f.write("unknown #{}\n".format(unknown))

#pad
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#Layers
embedding_dim = 50
'''
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
'''
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
#keras.utils.plot_model(model, show_shapes=True)

#logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#Training
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

history = model.fit(X_train, y_train, 
                    epochs=20, 
                    verbose=1, 
                    validation_data=(X_val, y_val), 
					callbacks=[tensorboard_callback],
                    batch_size=64)
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#model.save('Models/mod_3_erica_mixed')

ut.plot_history(history)
print(history.history)


# Save the weights we want to analyse as a variable. Note that the first value represents any 
# unknown word, which is not in the metadata, so we will remove that value.
weights = tf.Variable(model.layers[0].get_weights()[0][1:])
# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)
