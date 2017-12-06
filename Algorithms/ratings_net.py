
import sys
import tensorflow as tf
import numpy as np
from scipy.stats import invgauss
from scipy.special import erfc
from util import loadJSONForNet as loadJSON
from matplotlib import pyplot as plt

FLAGS = None

hidden_size = 300
dish = "Brownies"
input_file = "../dataProcessing/Processed Recipes/_" + dish + ".json"
activation_fn = tf.nn.sigmoid
max_epochs = 100
batch_size = 20

def undoLabelNormalization(all_data):
	mus = []
	sigmas = []
	for i in range(1):
		labels = all_data[1 + 2 * i]
		mu = np.mean(labels)
		sigma = np.std(labels)
		mus.append(mu)
		sigmas.append(sigma)

		modified_labels = (labels - mu) / sigma
		modified_labels = erfc(-modified_labels / 2)/2
		modified_labels = modified_labels * 4 + 1
		all_data[1 + 2 * i] = modified_labels
	return mus, sigmas

def batches(tensors, size, random=True):
	l = tensors[0].shape[0]
	if random:
		perm = np.random.permutation(l)
		tensors = [t[perm] for t in tensors]
	for batch in range(l/size):
		batch_data = tensors[0][size * batch: size * (batch + 1), :]
		batch_labels = tensors[1][size * batch: size * (batch + 1)]
		yield batch_data, batch_labels

def makeGraph(xs, ys, title, fn, setAxes=False):
	plt.figure()
	plt.scatter(xs, ys)
	plt.title(title)
	if setAxes:
		plt.xlim(1, 5)
		plt.ylim(1, 5)
	plt.savefig("../results/neural_net/" + dish + '/' + fn)

def train(data, labels):
	M = data.shape[0]
	N = data.shape[1]

	# Create the model
	x = tf.placeholder(tf.float32, [None, N])
	W1 = tf.Variable(tf.random_normal([N, hidden_size]))
	b1 = tf.Variable(tf.zeros([1, hidden_size]))

	h = activation_fn(tf.matmul(x, W1) + b1)
	W2 = tf.Variable(tf.random_normal([hidden_size, 1]))
	b2 = tf.Variable(tf.zeros(1))

	y_hat = tf.matmul(h, W2) + b2
	y = tf.placeholder(tf.float32, [None, 1])

	loss = tf.reduce_mean(tf.squared_difference(y, y_hat))
	train_step = tf.train.AdamOptimizer(0.1, 0.1).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Now, train
	epoch_losses = []
	for epoch in range(1, max_epochs + 1):
		batch_losses = []
		for batch_xs, batch_ys in batches([data, labels], batch_size):
			_, batch_loss, params = sess.run([train_step, loss, [W1, b1, W2, b2]], feed_dict={x: batch_xs, y: batch_ys})
			batch_losses.append(batch_loss)
		epoch_loss = sum(batch_losses) / len(batch_losses)
		epoch_losses.append(epoch_loss)
		if epoch % 10 == 0 and epoch > 0:
			print "Finished epoch %d, loss = %f" %(epoch, epoch_loss)

	makeGraph(range(max_epochs), epoch_losses, "Training loss", "train_loss")

	return params

def test(data, labels, params):
	M = data.shape[0]
	N = data.shape[1]

	# Create the model
	x = tf.placeholder(tf.float32, data.shape)
	W1 = tf.placeholder(tf.float32, [N, hidden_size])
	b1 = tf.placeholder(tf.float32, [1, hidden_size])

	h = activation_fn(tf.matmul(x, W1) + b1)
	W2 = tf.placeholder(tf.float32, [hidden_size, 1])
	b2 = tf.placeholder(tf.float32, [1])

	y_hat = tf.matmul(h, W2) + b2
	y = tf.placeholder(tf.float32, [M, 1])
	loss = tf.reduce_mean(tf.squared_difference(tf.cast(y, dtype=tf.float32), y_hat))

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	feed_dict = {x: data, y: labels, W1: params[0], b1: params[1], W2: params[2], b2: params[3]}
	result, predictions = sess.run([loss, y_hat], feed_dict=feed_dict)
	makeGraph(labels, predictions, "Labels vs. predictions", "testing", setAxes=True)
	print "Test loss: %f" % result

if __name__ == '__main__':
	all_data = list(loadJSON(input_file, [1, 1]))
	mus, sigmas = undoLabelNormalization(all_data)
	data_train, labels_train, data_dev, labels_dev, data_test, labels_test = all_data

	params = train(data_train, labels_train)
	test(data_train, labels_train, params)

