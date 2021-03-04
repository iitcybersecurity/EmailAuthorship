from gensim.models import Word2Vec
  
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
  
from sklearn import cluster
from sklearn import metrics

from preprocess.preprocess_text import *
from preprocess.utils_dataset import *
import sys
import json

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
  
# train data

target_user = "shackleton-s"
base_dir = "preprocess/RAW/enron_dataset/"
  
t_dataset = read_data(base_dir, target_user)
t_len = len([elem for elem in t_dataset])

#clean dataset

t_list = []
for email in t_dataset:
    new = preprocess(email)
    if(new):
        t_list.append(new)

#word2vec

sentences = t_list
 
model = Word2Vec(sentences, min_count=1)
  
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   
 
print ("========================")
#print (X)

#print (model[model.wv.vocab])

#print (model.similarity('tomorrow', 'eat'))
#print (model.most_similar(positive=['hallo'], negative=[], topn=2))


NUM_CLUSTERS=5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

print(str(assigned_clusters))
  

for index, sentence in enumerate(sentences):    
    print(str(assigned_clusters[index]) + ":" + str(sentence))
    if(index == 50):
        break
 

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
  
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
  
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
  
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
  
print ("Silhouette_score: ")
print (silhouette_score)
 
#plot 

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)
 
 
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
 
 
for j in range(len(sentences)):    
   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
   print ("%s %s" % (assigned_clusters[j],  t_dataset[j])) #sentences[j]
 
 
plt.show()