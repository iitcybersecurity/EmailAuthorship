from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from utils import *
import tensorflow as tf
import numpy as np
import cv2
from grad_cam_utils import *

#layerName = 'conv1d'

target_user = "shackleton-s"
base_dir = "Dataset/RAW/enron_dataset/"
file_tokenizer = 'token_rcnn_mixed.json'
file_model = 'rcnn_mod_mixed'
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
emails = pad_sequences(emails, padding='post', maxlen=50)
email_target = emails[0]
email_target = [tokenizer.index_word[elem] if elem > 0 else '_PAD_' for elem in email_target]

preds = model.predict(emails, verbose=1)
i = np.argmax(preds[0])
j = np.argmin(preds[0])

cam = GradCAM(model, i)
cam2 = GradCAM(model, j)

exp = [np.expand_dims(e, axis=0) for e in emails]
exp = np.array(exp)
emails = exp
heatmap = cam.compute_heatmap(emails[0])
heatmap2 = cam2.compute_heatmap(emails[0])

# resize the resulting heatmap to the original input image dimensions
heatmap = np.resize(heatmap, (1, 50))
heatmap2 = np.resize(heatmap2, (1, 50))

# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([emails[0], heatmap])
output2 = np.vstack([emails[0], heatmap2])

#print("vstack out: ", output, output.shape)
#output = imutils.resize(output, height=700)
cv2.imwrite('imgprova.jpg', output)

plot(range(50), heatmap[0], email_target)
plot(range(50), heatmap2[0], email_target)