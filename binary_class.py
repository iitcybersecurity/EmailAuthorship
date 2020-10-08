import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
import sys
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import numpy as np


sentences_train = tf.keras.preprocessing.text_dataset_from_directory(
    'Email',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=False,
    seed=False,
    validation_split=0.2, #Fraction of the training data to be used as validation data
    subset="training",
    follow_links=False,
)

sentences_val = tf.keras.preprocessing.text_dataset_from_directory(
    'Email',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=False,
    seed=False,
    validation_split=0.2, #Fraction of the training data to be used as validation data
    subset="validation",
    follow_links=False,
)


train=[str(element[0][0]) for element in sentences_train.as_numpy_iterator()]
val=[str(element[0][0]) for element in sentences_val.as_numpy_iterator()]

y_train=[int(element[1][0]) for element in sentences_train.as_numpy_iterator()]
y_val=[int(element[1][0]) for element in sentences_val.as_numpy_iterator()]


#Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train)

X_train = tokenizer.texts_to_sequences(train)
X_val = tokenizer.texts_to_sequences(val)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

#pad
maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

#Embedding layer
embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

#Training
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_val, y_val),
                    batch_size=10)
print(history.history)