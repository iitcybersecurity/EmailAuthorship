from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from utils import *
import tensorflow as tf
import numpy as np
import cv2
from grad_cam_utils import *

maxlen = 300

target_user = "shackleton-s"
base_dir = "Dataset/RAW/enron_dataset/"
file_tokenizer = 'Tokenizer/token_{}.json'.format(target_user)
file_model = target_user
file_labels_test = "label_test_{}.txt".format(target_user)
file_test = "test_{}.txt".format(target_user)

print("[LOADING Model] : ", file_model)
model = keras.models.load_model('Models/' + file_model)
print("[LOADING Weigths]")
weights = tf.Variable(model.layers[0].get_weights()[0][1:])
print("[LOADING Tokenizer]")
with open(file_tokenizer) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

t_dataset, t_labels, nt_dataset, nt_labels = read_data(base_dir, target_user)
t_len = len([elem for elem in t_labels])
nt_len = len([elem for elem in nt_labels])

#Shuffle dataset
nt_dataset_shuffled, nt_labels_shuffled = shuffle_data(nt_dataset, nt_labels)
dataset = t_dataset + nt_dataset[0:t_len]
labels = t_labels + nt_labels[0:t_len]
dataset_shuffled, labels_shuffled = shuffle_data(dataset, labels)

emails = tokenizer.texts_to_sequences(dataset_shuffled)
emails = pad_sequences(emails, padding='post', maxlen=maxlen)
email_target = emails[0] #scelta arbitraria
email_target = [tokenizer.index_word[elem] if elem > 0 else '_PAD_' for elem in email_target]

index_0 = -1
index_1 = -1
for i in range(len(labels_shuffled)):
    if(index_0>=0 and index_1>=0):
        break
    if(labels_shuffled[i] == 0 and index_0 < 0):
        index_0 = i
    if(labels_shuffled[i] == 1 and index_1 < 0):
        index_1 = i

email = emails[0] #prova
email = [tokenizer.index_word[elem] if elem > 0 else '_PAD_' for elem in email]
email_0 = emails[index_0]
email_0 = [tokenizer.index_word[elem] if elem > 0 else '_PAD_' for elem in email_0]
email_1 = emails[index_1]
email_1 = [tokenizer.index_word[elem] if elem > 0 else '_PAD_' for elem in email_1]

preds = model.predict(emails, verbose=1)
print(preds)

i = np.argmax(preds[0]) #preds[0] perche' email traget email[0]
j = np.argmin(preds[0])
print("i: ", preds[0][i], "j: ", preds[0][j])
i_0 = np.argmax(preds[index_0])
i_1 = np.argmax(preds[index_1])

print("Predicted: ", i, "Class: ", labels_shuffled[0])
#print("Predicted: ", i_0, "Class: ", labels_shuffled[index_0])
#print("Predicted: ", i_1, "Class: ", labels_shuffled[index_1]) #target

print(model.summary())
cam = GradCAM(model, i)
print(cam.layerName)
cam2 = GradCAM(model, j)
print(cam2.layerName)

'''
cam_0 = GradCAM(model, i_0)
cam_1 = GradCAM(model, i_1)
'''
exp = [np.expand_dims(e, axis=0) for e in emails]
exp = np.array(exp)
emails = exp
heatmap = cam.compute_heatmap(emails[0])
heatmap2 = cam2.compute_heatmap(emails[0])
'''
heatmap_0 = cam_0.compute_heatmap(emails[index_0])
heatmap_1 = cam_1.compute_heatmap(emails[index_1])
'''
# resize the resulting heatmap to the original input image dimensions
heatmap = np.resize(heatmap, (1, maxlen))
heatmap2 = np.resize(heatmap2, (1, maxlen))
'''
heatmap_0 = np.resize(heatmap_0, (1, 50))
heatmap_1 = np.resize(heatmap_1, (1, 50))
'''
# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([emails[0], heatmap, heatmap2])
#print("out", output)
'''
output = np.vstack([emails[0], heatmap])
output_0 = np.vstack([emails[index_0], heatmap_0])
output_1 = np.vstack([emails[index_1], heatmap_1])
print("vstack out_0: ", output_0, output_0.shape)
print("vstack out_1: ", output_1, output_1.shape)
#cv2.imwrite('imgprova.jpg', output)
'''
#plot(range(50), heatmap[0], email_target)
plot(range(50), heatmap[0][0:50], email[0:50]) #neg
plot(range(50), heatmap2[0][0:50], email[0:50]) #pos

