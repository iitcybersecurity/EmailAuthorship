from gensim.models import Word2Vec
  
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
  
from sklearn import cluster
from sklearn import metrics

from preprocess.preprocess_text import *
from preprocess.utils_dataset import *
import sys
import json, os, math

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
  
# train data

target_user = "shackleton-s"
non_target = "dasovich-j"

base_dir = "preprocess/RAW/enron_dataset/"
log_dir =  "preprocess/email_csv/"

t_dataset = []

'''
with open("preprocess/USERS.txt", "r") as users:
    for u in users:
        target_user = u.replace('\n', '')
        t_dataset += read_data(base_dir, target_user)
'''

t_dataset = read_data(base_dir, target_user)
nt_dataset = read_data(base_dir, non_target)

num_samples = min(len(t_dataset), len(nt_dataset))

user = [0 for i in range(num_samples)]
user1 = [1 for i in range(num_samples)]

print(target_user, len(user))
print(non_target, len(user1))

user += user1

#clean dataset

t_list = []
nt_list = []
for email in t_dataset:
    new = preprocess(email) #text_to_word_list(email) #
    if(new):
        t_list.append(new)

for email in nt_dataset:
    new = preprocess(email) #text_to_word_list(email) #
    if(new):
        nt_list.append(new)

#word2vec

sentences = t_list[:num_samples] + nt_list[:num_samples]

 
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


NUM_CLUSTERS=10
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

#print(str(assigned_clusters))
  
'''
for index, sentence in enumerate(sentences):    
    print(str(assigned_clusters[index]) + ":" + str(sentence))
    if(index == 50):
        break
'''

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

split = int(len(sentences)/6)
print("split", split) 
cluster_csv = os.path.join(log_dir, '{}.csv'.format(target_user)) 
with open(cluster_csv, 'w') as f:
    f.write("Sent1,Sent2,Same\n")
    for j in range(len(sentences)):
        plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
        same0 = []
        same1 = []
        is_same = 1
        
        if(j < 2*split):
            #same = 1
            if(j in range(0, split)):
                sent0 = ' '.join(sentences[j])
                sent1 = ' '.join(sentences[j+split])

            #same = 0
            if(j in range(split, 2*split)):
                sent0 = ' '.join(sentences[j+split])
                sent1 = ' '.join(sentences[j+split*2])
                is_same = 0

            f.write("{},{},{}\n".format(sent0, sent1, is_same))
        

'''    
    f.write("User\tCluster\tSentence\n")
    for j in range(len(sentences)):    
        plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
        #file csv | Topic | Sentence |
        sent = ' '.join(sentences[j])
        f.write("{}\t{}\t{}\n".format(user[j], assigned_clusters[j], sent)) #print ("%s %s" % (assigned_clusters[j],  t_dataset[j])) #sentences[j] 
'''

plt.savefig('plt.png') 
#plt.show()

#csv file | sentence1 | sentence2 | is_equal |

