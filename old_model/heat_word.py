from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from utils import *
from heat_word_utils import *
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from grad_cam_utils import *

maxlen = 300

target_user = "shackleton-s"
base_dir = "Dataset/RAW/enron_dataset/"
file_tokenizer = 'Tokenizer/token_{}.json'.format(target_user)
file_tokenizer_not_target = 'Tokenizer/not_target_{}.json'.format(target_user)
file_model = target_user
print("[LOADING Model] : ", file_model)
model = keras.models.load_model('Models/' + file_model)
print("[LOADING Weigths]")
weights = tf.Variable(model.layers[0].get_weights()[0][1:])
print("[LOADING Tokenizer]")
with open(file_tokenizer) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
with open(file_tokenizer_not_target) as f:
    data = json.load(f)
    tokenizer_not = tokenizer_from_json(data)

save_word_dict("Word_dict_t.tsv", tokenizer.word_index)
save_word_dict("Word_dict_nt.tsv", tokenizer_not.word_index)

t_dataset, t_labels, nt_dataset, nt_labels = read_data(base_dir, target_user)
t_len = len([elem for elem in t_labels])
nt_len = len([elem for elem in nt_labels])

#Shuffle dataset
nt_dataset_shuffled, nt_labels_shuffled = shuffle_data(nt_dataset, nt_labels)
dataset = t_dataset + nt_dataset[0:t_len]
labels = t_labels + nt_labels[0:t_len]
dataset_shuffled, labels_shuffled = shuffle_data(dataset, labels)
DIM = len(labels_shuffled)

emails = tokenizer.texts_to_sequences(dataset_shuffled)
emails = pad_sequences(emails, padding='post', maxlen=maxlen)
email_target = emails[0]
email_target = [tokenizer.index_word[elem] if elem > 0 else '_PAD_' for elem in email_target]

preds = model.predict(emails, verbose=1)
arr_i = [np.argmax(preds[idx]) for idx in range(DIM)]

arr_cam = [GradCAM(model, arr_i[idx]) for idx in range(DIM)]
exp = [np.expand_dims(e, axis=0) for e in emails]
exp = np.array(exp)
emails = exp

arr_heatmap = []
for i in range(DIM):
    arr_heatmap.append(arr_cam[i].compute_heatmap(emails[i])) #it takes looooong

arr_heatmap = list(map(lambda h: np.resize(h, (1,maxlen)), arr_heatmap))

with open('{}_heatmap.tsv'.format(target_user), 'w') as f:
    f.write("Word\tHeat\tClass\n")
    for i in range(DIM):
        j = 0
        for word in emails[i][0]:
            if(word == 0):
                j += 1
                continue
            str_w = tokenizer.index_word[word]
            heat = arr_heatmap[i][0][j]
            f.write("{}\t{}\t{}\n".format(str_w, heat, labels_shuffled[i]))
            j += 1
sort_csv(target_user)
words_t, words_nt, heats_t, heats_nt, labels = split_csv(target_user)

common_words, heats_1, heats_0 = words_in_common(target_user)

#print("heatmaps", arr_heatmap[0][0].shape, heats_t.shape, heats_nt.shape)
horizontal_plot(common_words[100:150], heats_1[100:150], heats_0[100:150])
'''
plot(range(50), heats_t[:50], words_t[:50])
plot(range(50), heats_nt[:50], words_nt[:50])
'''

'''
output = np.vstack([emails[0], heatmap])
print("vstack out: ", output, output.shape)
plot(range(50), heatmap[0], email_target)
'''
