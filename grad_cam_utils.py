from tensorflow.keras.models import Model
from utils import *
import tensorflow as tf
import numpy as np
import cv2

def plot(indexes, heats, words):
	plt.figure(figsize=(12, 5))
	plt.bar(indexes, heats)
	for i in indexes:
		plt.text(i-.25, max(0, heats[i]*1.05), words[i], rotation=90)
	plt.title('Grad-cam')
	plt.show()

class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 3D output
			if len(layer.output_shape) == 3:
				return layer.name
		# otherwise, we could not find a 3D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 3D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, mail, eps=1e-8):
		# construct our gradient model by supplying 
		# (1) the inputs to our pre-trained model, 
		# (2) the output of the (presumably) final 3D layer in the network, and 
		# (3) the output of the softmax activations from the model
		gradModel = Model(
					inputs=[self.model.inputs],
					outputs=[self.model.get_layer(self.layerName).output,
							 self.model.output])
							 
		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the mail tensor to a float-32 data type, pass the
			# mail through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(mail, tf.float32)

			(convOutputs, predictions) = gradModel(inputs)
			print("conv: ", convOutputs, "\n pred: ", predictions)
			loss = predictions[:, self.classIdx]
			print("loss: ", loss)
			#print(loss)
		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)
		print("Grads: ", grads)

		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# grab the spatial dimensions of the input mail and resize
		# the output class activation map to match the input mail
		# dimensions
		(w, h) = (mail.shape[1], 1)
		heatmap = cv2.resize(cam.numpy(), (h, w))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
		return heatmap
