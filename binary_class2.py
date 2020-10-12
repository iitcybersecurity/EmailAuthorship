import utils as ut
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing.sequence import pad_sequences



#shackleton-s => target
#dasovich-j => other

sentences_train = tf.keras.preprocessing.text_dataset_from_directory(
    'Email_train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=True,
    seed=3210,
    validation_split=0.16, #Fraction of the training data to be used as validation data
    subset="training",
    follow_links=False,
)

sentences_val = tf.keras.preprocessing.text_dataset_from_directory(
    'Email_train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=1,
    max_length=None,
    shuffle=True,
    seed=3210,
    validation_split=0.16, #Fraction of the training data to be used as validation data
    subset="validation",
    follow_links=False,
)

sentences_test = tf.keras.preprocessing.text_dataset_from_directory(
    'Email_test',
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
tokenizer = Tokenizer(num_words=500)
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
maxlen = 20

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#Layers
embedding_dim = 10

model = Sequential()
model.add(layers.Embedding(input_dim=500, 
                           output_dim=embedding_dim, 
                           input_length=maxlen,
                           mask_zero=True)) #masks zeroes added in padding
model.add(layers.Flatten())
model.add(layers.Dropout(0.60))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
#keras.utils.plot_model(model, show_shapes=True)    


#Training
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)
#print(len(y_test))

history = model.fit(X_train, y_train,
                    epochs=40,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    batch_size=32)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#Print prediction samples
#pred = model.predict_classes(X_test, verbose = 2)
#for i in range(len(y_test)-1) :
#    print('Expected:', y_train[i], 'Predicted', pred[i]) 

ut.plot_history(history)
print(history.history)
