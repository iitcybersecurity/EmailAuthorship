import json, os, math
import csv
from utils import *
from keras_models import *
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

token_flag = True #save token
flag_model = True #save model
maxlen = 300
embedding_dim = 20
epochs = 15

if len(sys.argv) < 2 :
  print("Specifica Target user")
  sys.exit()

val = sys.argv[1]
print(val)

target_user = val#"shackleton-s"
base_dir = "Dataset/RAW/enron_dataset/"
log_dir = "Models/logs/{}/".format(target_user)
file_labels_test = "label_test_{}.txt".format(target_user)
file_test = "test_{}.txt".format(target_user)

#Data for saving
file_tokenizer = 'Tokenizer/token_{}.json'.format(target_user)
file_model = target_user

#Read data from dataset
t_dataset, t_labels, nt_dataset, nt_labels = read_data_2050(base_dir, target_user, 0, 0)
t_len = len([elem for elem in t_labels])
nt_len = len([elem for elem in nt_labels])

#Shuffle dataset
nt_dataset_shuffled, nt_labels_shuffled = shuffle_data(nt_dataset, nt_labels)
dataset = t_dataset + nt_dataset[0:t_len]
labels = t_labels + nt_labels[0:t_len]
dataset_shuffled, labels_shuffled = shuffle_data(dataset, labels)

#Get 80% training, 10% validation, 10% testing
train, val, test = split_dataset(dataset_shuffled, 0.8, 0.1, 0.1, file_test)
y_train, y_val, y_test = split_dataset(labels_shuffled, 0.8, 0.1, 0.1, file_labels_test)

print("Training samples: " + str(len(y_train)))
print("Val samples: " + str(len(y_val)))
print("Testing samples: " + str(len(y_test)))

#Tokenizer
tokenizer = Tokenizer(num_words=16000)
tokenizer.fit_on_texts(train)
tokenizer.fit_on_texts(val)
tokenizer.fit_on_texts(test)

#save tokenizer
save_tokenizer_in(token_flag, file_tokenizer , tokenizer)

X_train = tokenizer.texts_to_sequences(train)
X_val = tokenizer.texts_to_sequences(val)
X_test = tokenizer.texts_to_sequences(test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
word_index = tokenizer.word_index
subwords = word_index

# Save Labels separately on a line-by-line manner.
filename = os.path.join(log_dir, '{}_metadata.tsv'.format(target_user))
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open(filename, 'w') as f:
  f.write("Word\tLabel\n")
  for key, value in subwords.items():
    f.write("{}\t{}\n".format(str(key), value))
  # Fill in the rest of the labels with "unknown"
  #for unknown in range(1, vocab_size - len(subwords)):
  #	f.write("unknown #{}\n".format(unknown))

#pad
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#Layers
embedding_matrix = create_embedding_matrix(
                    'pretrained_Glove/glove.6B.100d.txt',
                    word_index, 
                    vocab_size,
                    embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

model = cnn_embedding_model(vocab_size, embedding_dim, maxlen, matrix = embedding_matrix)#cnn_embedding_model()
model.summary()

#logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Training
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    verbose=1, 
                    validation_data=(X_val, y_val), 
                    callbacks=[tensorboard_callback],
                    batch_size=64)
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training Loss: {:.4f}".format(loss))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing Loss: {:.4f}".format(loss))
if(flag_model):
  model.save('Models/' + file_model)

img_path = 'Models/{}/plot.png'.format(file_model)
plot_history(history, img_path)
#print(history.history)

# Save weights
weights = tf.Variable(model.layers[0].get_weights()[0][1:])

# Create a checkpoint from embedding
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

# %tensorboard --logdir Models/logs/first_mixed 
# => Projector => Load "metadata.tsv"
