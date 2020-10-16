import json
import utils as ut
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences



#shackleton-s => target
#mix => other

sentences_train = tf.keras.preprocessing.text_dataset_from_directory(
    'Dataset/Email_train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=False,
    seed=3210,
    validation_split=0.16, #Fraction of the training data to be used as validation data
    subset="training",
    follow_links=False,
)

sentences_val = tf.keras.preprocessing.text_dataset_from_directory(
    'Dataset/Email_train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=False,
    seed=3210,
    validation_split=0.16, #Fraction of the training data to be used as validation data
    subset="validation",
    follow_links=False,
)

sentences_test = tf.keras.preprocessing.text_dataset_from_directory(
    'Dataset/Email_test',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=False,
    seed=None,
    validation_split=None, #Fraction of the training data to be used as validation data
    subset=None,
    follow_links=False,
)

train=[str(element[0][0].decode('UTF-8')) for element in sentences_train.as_numpy_iterator()]
val=[str(element[0][0].decode('UTF-8')) for element in sentences_val.as_numpy_iterator()]
test=[str(element[0][0].decode('UTF-8')) for element in sentences_test.as_numpy_iterator()]

y_train=[int(element[1][0]) for element in sentences_train.as_numpy_iterator()]
y_val=[int(element[1][0]) for element in sentences_val.as_numpy_iterator()]
y_test=[int(element[1][0]) for element in sentences_test.as_numpy_iterator()]


#Tokenizer
tokenizer = Tokenizer(num_words=600)
tokenizer.fit_on_texts(train)
tokenizer.fit_on_texts(val)
tokenizer.fit_on_texts(test)


X_train = tokenizer.texts_to_sequences(train)
X_val = tokenizer.texts_to_sequences(val)
X_test = tokenizer.texts_to_sequences(test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
word_index = tokenizer.word_index

#with open('word_index.txt', 'w') as f:
#   print(word_index, file=f)

#pad
maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#Embedding matrix
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

embedding_dim = 7
embedding_matrix = create_embedding_matrix(
                    'pretrained_Glove/glove.6B.50d.txt',
                    word_index, 
                    vocab_size,
                    embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
#print(nonzero_elements / vocab_size)



#Layers
model = Sequential()
model.add(layers.Embedding(vocab_size, 
                           embedding_dim, 
                           #weights=[embedding_matrix], 
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
keras.utils.plot_model(model, show_shapes=True)


#Training
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
val_labels = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

shuffler = np.random.permutation(len(X_train))
X_train = X_train[shuffler]
y_train = y_train[shuffler]

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

history = model.fit(X_train, y_train,
                    epochs=30,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    batch_size=32)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

confusion_matrix = np.zeros((2, 2))
pred_labels = model.predict(X_val)
for i in range(0, len(pred_labels)):
    clas = 1
    if pred_labels[i][0] > pred_labels[i][1]:
        clas = 0
    confusion_matrix[clas][val_labels[i]] += 1

confusion_matrix /= np.sum(confusion_matrix)
print(confusion_matrix)


ut.plot_history(history)
print(history.history)
