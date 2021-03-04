import json, os, math
import pandas as pd
from utils import *
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

target = "nemec-g"#"perlingiere-d"#"dasovich-j"#"shackleton-s"
base_dir = "Dataset/RAW/enron_dataset/"
maxlen = 300

#Data for saving
file_tokenizer = 'Tokenizer/token_{}.json'.format(target)
file_tok_not_target = 'Tokenizer/not_target_{}.json'.format(target)
file_model = target

#load tokenizer + model
with open(file_tokenizer) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

#tokenizer non_target
#not_target_tokenizer = Tokenizer(num_words=16000)
model = keras.models.load_model('Models/' + file_model)

file_tsv = 'test_models/' + file_model
with open(file_tsv, 'w') as f:
    f.write("Accuracy\tID\tN_Emails\tW_Avg\n")

for filename in os.listdir(base_dir):
    '''
    if(filename == (target + ".json")):
        continue
    '''
    with open(base_dir + filename) as filecontent:
        emails = []
        not_target = []
        labels = []
        num_email = 0
        avg_words = 0
        data = json.load(filecontent)
        msgs = data['sent']
        for msg in msgs:
            body = msg['body']
            if(body == ""):
                continue
            num_email+=1
            emails.append(body)
            labels.append(1 if data['identity'] == target else 0)
        #not_target_tokenizer.fit_on_texts(not_target)
        emails = tokenizer.texts_to_sequences(emails)
        count = [len(elem) for elem in emails]
        avg_words = round(Average(count))
        emails = pad_sequences(emails, padding='post', maxlen=maxlen)
        emails = np.array(emails)
        labels = np.array(labels)
        labels = to_categorical(labels, 2)

        preds = model.predict(emails)
        
        loss, accuracy = model.evaluate(emails, labels, verbose=False)
        
        with open(file_tsv, 'a') as f:
            f.write("{:.4f}\t{}\t{}\t{}\n".format(accuracy, data['identity'], num_email, avg_words))

        print("Training Accuracy: {:.4f} User: {}".format((accuracy), data['identity']))

f = pd.read_csv(file_tsv, delimiter="\t", index_col=False)
identity = [f['ID'][i] for i in range(len(f))]
Accuracy = [f['Accuracy'][i] for i in range(len(f))]
weight = [f['N_Emails'][i] for i in range(len(f))]

print("Media pesata: ", np.average(Accuracy, weights=weight), "Media: ", np.average(Accuracy))
#save_tokenizer_in(True, file_tok_not_target , not_target_tokenizer)
#save_word_dict("nontarget.txt", not_target_tokenizer.word_index)

        


