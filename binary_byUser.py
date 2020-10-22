import json, os, math
import utils as ut
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.models import Sequential
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def Average(lst): 
    return sum(lst) / len(lst) 

target = "shackleton-s"
base_dir = "Dataset/RAW/enron_dataset/"

#Data for saving
file_tokenizer = 'token_first_mixed.json'
file_model = 'first_mod_mixed'

#load tokenizer + model
with open(file_tokenizer) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

model = keras.models.load_model('Models/' + file_model)

for filename in os.listdir(base_dir):
    with open(base_dir + filename) as filecontent:
        emails = []
        labels = []
        num_email = 0
        avg_words = 0
        data = json.load(filecontent)
        msgs = data['sent']
        for msg in msgs:
            body = msg['body']
            if body == "":
                continue

            if len(body) > 500:
                continue
            num_email+=1
            emails.append(body)
            labels.append(1 if data['identity'] == target else 0)

        emails = tokenizer.texts_to_sequences(emails)
        count = [len(elem) for elem in emails]
        avg_words = round(Average(count))
        emails = pad_sequences(emails, padding='post', maxlen=50)
        emails = np.array(emails)
    
        labels = np.array(labels)
        labels = to_categorical(labels, 2)

        loss, accuracy = model.evaluate(emails, labels, verbose=False)
        with open('test_models/' + file_model, 'a') as f:
            f.write(format(accuracy) + " " + 
            data['identity'] + " " + 
            format(num_email) +
            " " +
            format(avg_words) +
            "\n")
        print("Training Accuracy: {:.4f} User: {}".format((accuracy), data['identity']))
        


