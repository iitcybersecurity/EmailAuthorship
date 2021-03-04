from tensorflow.keras.models import Model
from utils import *
from grad_cam_utils import *
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd

f = pd.read_csv("final_heatmap_shackleton-s.tsv", delimiter=",")
grp = f.groupby(["Word"])["Class"].count()
grp.to_csv("prova.tsv")
f2 = pd.read_csv("prova.tsv", delimiter=",")
f2 = f2[f2["Class"]==2]
f2.to_csv("prova.tsv")
f2 = pd.read_csv("prova.tsv")
merge = f.merge(f2, on="Word")
common_words = [merge["Word"][i] for i in range(len(merge)) if merge["Class_x"][i]==1]
heats_1 = [merge["Heat"][i] for i in range(len(merge)) if merge["Class_x"][i]==1]
heats_0 = [merge["Heat"][i] for i in range(len(merge)) if merge["Class_x"][i]==0]

horizontal_plot(common_words[:15], heats_1[:15], heats_0[:15])

def words_in_common(target):
    f = pd.read_csv("final_heatmap_{}.tsv".format(target), delimiter=",")
    grp = f.groupby(["Word"])["Class"].count()
    grp.to_csv("prova.tsv")
    f2 = pd.read_csv("prova.tsv", delimiter=",")
    f2 = f2[f2["Class"]==2]
    f2.to_csv("prova.tsv")
    f2 = pd.read_csv("prova.tsv")
    merge = f.merge(f2, on="Word")
    common_words = [merge["Word"][i] for i in range(len(merge)) if merge["Class_x"][i]==1]
    heats_1 = [merge["Heat"][i] for i in range(len(merge)) if merge["Class_x"][i]==1]
    heats_0 = [merge["Heat"][i] for i in range(len(merge)) if merge["Class_x"][i]==0]
    return common_words, np.array(heats_1), np.array(heats_0)

def split_csv(target):
    f = pd.read_csv("final_heatmap_{}.tsv".format(target), delimiter=",")
    words_target = [f["Word"][i] for i in range(len(f)) if f["Class"][i] == 1]
    words_nt = [f["Word"][i] for i in range(len(f)) if f["Class"][i] == 0]

    #words = list(map(lambda w: tokenizer.word_index[w], words))
    heats_target = [f["Heat"][i] for i in range(len(f)) if f["Class"][i] == 1]
    heats_nt = [f["Heat"][i] for i in range(len(f)) if f["Class"][i] == 0]

    labels = [f["Class"][i] for i in range(len(f))]

    return words_target, words_nt, np.array(heats_target), np.array(heats_nt), np.array(labels)


#Sort csv
def sort_csv(target):
    file_tsv = pd.read_csv("{}_heatmap.tsv".format(target), delimiter="\t")
    group = file_tsv.groupby(["Word", "Class"], sort=False)["Heat"].mean()
    group.to_csv("group_heatmap_{}.tsv".format(target))
    f = pd.read_csv("group_heatmap_{}.tsv".format(target), delimiter=",")
    sort_group = f.sort_values(["Heat"], ascending=False)
    sort_group.to_csv("final_heatmap_{}.tsv".format(target), index=False)

