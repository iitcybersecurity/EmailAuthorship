import pandas as pd
import numpy as np
import sys
from konlpy.tag import Mecab
from collections import Counter
import operator
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn


import json, os, math
import csv
from utils import *
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
base_dir = "Dataset/RAW/enron_dataset/"
log_dir = "Models/logs/first_mixed/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#Data for saving
file_tokenizer = 'token_first_mixed.json'
file_model = 'first_mod_mixed'

#Read data from dataset
t_dataset, t_labels, nt_dataset, nt_labels = read_data(base_dir, target_user)
t_len = len([elem for elem in t_labels])
nt_len = len([elem for elem in nt_labels])

#Shuffle dataset
nt_dataset_shuffled, nt_labels_shuffled = shuffle_data(nt_dataset, nt_labels)
dataset = t_dataset + nt_dataset[0:t_len]
labels = t_labels + nt_labels[0:t_len]
dataset_shuffled, labels_shuffled = shuffle_data(dataset, labels)
train, val, test = split_dataset(dataset_shuffled, 0.8, 0.1, 0.1)
y_train, y_val, y_test = split_dataset(labels_shuffled, 0.8, 0.1, 0.1)

# %matplotlib inline
mecab = Mecab()

tbl = {
    'id': np.arange(len(np.array(train))),
    'data': np.array(train),
    'labels': np.array(y_train)
}

tbl_test = {
    'id': np.arange(len(np.array(test))),
    'data': np.array(test),
    'labels': np.array(y_test)
}

tbl = pd.DataFrame.from_dict(tbl)
tbl_test = pd.DataFrame.from_dict(tbl_test)

keywords = [mecab.morphs(str(i).strip()) for i in tbl['data']]

print(np.median([len(k) for k in keywords]), len(keywords), tbl.shape)

keyword_cnt = Counter([i for item in keywords for i in item])

#using top  5,000 keywords. 
keyword_clip = sorted(keyword_cnt.items(), key=operator.itemgetter(1))[-10000:]
keyword_clip_dict = dict(keyword_clip)
keyword_dict = dict(zip(keyword_clip_dict.keys(), range(len(keyword_clip_dict))))

#for paddning and unknown keywords  
keyword_dict['_PAD_'] = len(keyword_dict)
keyword_dict['_UNK_'] = len(keyword_dict) 

#Create dictionary to backtrack keywords
keyword_rev_dict = dict([(v,k) for k, v in keyword_dict.items()])

#decide max sequence length
max_seq =np.median([len(k) for k in keywords]) + 5


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """
    from keras
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def encoding_and_padding(corp_list, dic, max_seq=50):
    coding_seq = [ [dic.get(j, dic['_UNK_']) for j in i]  for i in corp_list ]
    #In general, reviews are likely to include a lot of information at the end, so pre-padding is preceded.
    return(pad_sequences(coding_seq, maxlen=max_seq, padding='pre', truncating='pre',value=dic['_PAD_']))


train_x = encoding_and_padding(keywords, keyword_dict, max_seq=int(max_seq))
train_y = tbl['labels']
#print(train_x.shape, train_y.shape)

# split train, test 
tr_idx = np.random.choice(train_x.shape[0], int(train_x.shape[0] * 0.9) )
tr_idx_set = set(tr_idx)
te_idx = np.array([i for i in range(train_x.shape[0]) if i not in tr_idx_set])
print(len(te_idx), len(tr_idx))

tr_set = gluon.data.ArrayDataset(train_x[tr_idx], train_y[tr_idx].to_numpy())
tr_data_iterator = gluon.data.DataLoader(tr_set, batch_size=100, shuffle=True)

te_set =gluon.data.ArrayDataset(train_x[te_idx], train_y[te_idx].to_numpy())
te_data_iterator = gluon.data.DataLoader(te_set, batch_size=100, shuffle=True)

#print(type(train_y), type(train_x), type(tr_data_iterator))

#Making Model
class SentClassificationModel(gluon.Block):

    def __init__(self, vocab_size, num_embed, **kwargs):
        super(SentClassificationModel, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(input_dim=vocab_size, output_dim=num_embed,
                                        weight_initializer = mx.init.Xavier())
            self.conv = nn.Conv1D(128, 5, activation='relu')
            self.dropout = nn.Dropout(rate=0.2)
            self.pool = nn.GlobalMaxPool1D()
            self.dense1 = nn.Dense(units=10, activation='relu')
            self.dense2 = nn.Dense(units=2, activation='sigmoid')

            self.conv1_act = None

                        
    def forward(self, inputs, get_act=False):
        em_out = self.embed(inputs)
        #print(em_out)
        em_swaped = mx.nd.swapaxes(em_out, 1,2)
        conv_ = self.conv(em_swaped)
        #print(conv_)
        if get_act:
            self.conv1_act = conv_
        dropout_ = self.dropout(conv_)
        pool_ = self.pool(dropout_)
        #print("Dropout: ", dropout_)
        #print("Pool: ", pool_.shape)
        dense1_ = self.dense1(pool_)
        #print("Dense1: ", dense1_.shape)
        outs = self.dense2(dense1_)
        #print("Outs: ", outs.shape)

        return outs

ctx = mx.cpu()

model = SentClassificationModel(vocab_size = len(keyword_dict), num_embed=50)
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), 'sgd', optimizer_params={'learning_rate':0.01, 'wd':0.00001 })
loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

print(model)

#Training
def calculate_loss(model, data_iter, loss_obj, ctx=ctx):
    test_loss = []
    for i, (te_data, te_label) in enumerate(data_iter):
        te_data = te_data.as_in_context(ctx)
        te_label = te_label.as_in_context(ctx)
        with autograd.predict_mode():
            te_output = model(te_data)
            loss_te = loss_obj(te_output, mx.nd.one_hot(te_label, 2))
        curr_loss = mx.nd.mean(loss_te).asscalar()
        test_loss.append(curr_loss)
    return(np.mean(test_loss))

epochs = 15

tot_test_loss = []
tot_train_loss = []

for e in range(epochs):
    train_loss = []
    #batch training 
    for i, (data, label) in enumerate(tr_data_iterator):
        #print("Data: ", data.shape)
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record(train_mode=False):
            output = model(data)
            loss_ = loss(output, mx.nd.one_hot(label, 2))
            loss_.backward()
        trainer.step(data.shape[0])
        curr_loss = mx.nd.mean(loss_).asscalar()
        train_loss.append(curr_loss)

    #caculate test loss
    test_loss = calculate_loss(model, te_data_iterator, loss_obj = loss, ctx=ctx) 

    print("Epoch %s. Train Loss: %s, Test Loss : %s" % (e, np.mean(train_loss), test_loss))    
    tot_test_loss.append(test_loss)
    tot_train_loss.append(np.mean(train_loss))

plt.plot(tot_train_loss)
plt.plot(tot_test_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


def grad_cam_conv1D(model, x, loss, ctx):
    with autograd.record():
        output = model.forward(mx.nd.array([x,],ctx=ctx),get_act=True)
        loss_ = loss(output, mx.nd.array([[0.0,1.0],],ctx=ctx))
        output = mx.nd.SoftmaxActivation(output)
        print(output)
        loss_.backward()
    acts = model.conv1_act
    pooled_grad = mx.nd.mean(model.conv.weight.grad(), axis=(1,2))
    print(pooled_grad)
    for i in range(acts.shape[1]):
        acts[:,i,:] *= pooled_grad[i]
    heat = mx.nd.mean(acts, axis=1)
    return(heat.asnumpy()[0][1:-1], loss_)

#making test set 
parsed_text = [mecab.morphs(str(i).strip()) for i in tbl_test['data']]

test_x = encoding_and_padding(parsed_text, keyword_dict, max_seq=int(max_seq))
test_y = tbl_test['labels']

prob = model(mx.nd.array(test_x, ctx=ctx))

prob_np = mx.nd.SoftmaxActivation(prob).asnumpy()

prob_np_a = prob_np[:,1]
'''
#roc tells it's quite good performance
require(pROC)
plot(roc(test_y, prob_np_a), print.auc==TRUE)
'''
idx = 2

# below review means "Only a few masterpieces are full of rubbish." with negative feedback.
prob_np[idx], tbl_test.iloc[idx], test_y[idx]

heat, loss__ = grad_cam_conv1D(model, test_x[idx], loss=loss, ctx=ctx)

hm_tbl = pd.DataFrame({
    'heat':heat, 
    'kw':[keyword_rev_dict[i] for i in test_x[idx][:len(heat)] ]
    })

plot(hm_tbl.index, hm_tbl['heat'], hm_tbl['kw'])
hm_tbl.to_csv('hm_tbl')


