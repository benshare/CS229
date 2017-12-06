
import sys
import tensorflow as tf
import numpy as np
from util import loadJSONForNet as loadJSON

FLAGS = None

hidden_size = 300
input_file = "../data/_Brownies.json"
activation_fn = tf.nn.sigmoid
max_epochs = 100
batch_size = 20

def batches(tensors, size, random=True):
	l = tensors[0].shape[0]
	if random:
		perm = np.random.permutation(l)
		tensors = [t[perm] for t in tensors]
	for batch in range(l/size):
		batch_data = tensors[0][size * batch: size * (batch + 1), :]
		batch_labels = tensors[1][size * batch: size * (batch + 1)]
		yield batch_data, batch_labels

def train(data, labels):
	M = data.shape[0]
	N = data.shape[1]

	# Create the model
	x = tf.placeholder(tf.float32, [None, N])
	W1 = tf.Variable(tf.zeros([N, hidden_size]))
	b1 = tf.Variable(tf.zeros([1, hidden_size]))

	h = activation_fn(tf.matmul(x, W1) + b1)
	W2 = tf.Variable(tf.zeros([hidden_size, 1]))
	b2 = tf.Variable(tf.zeros(1))

	y_hat = tf.matmul(h, W2) + b2
	y = tf.placeholder(tf.float32, [None, 1])

	loss = tf.reduce_mean(tf.squared_difference(y, y_hat))
	train_step = tf.train.AdamOptimizer(0.1, 0.1).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Now, train
	losses = []
	for epoch in range(1, max_epochs + 1):
		for batch_xs, batch_ys in batches([data, labels], batch_size):
			_, batch_loss, params = sess.run([train_step, loss, [W1, b1, W2, b2]], feed_dict={x: batch_xs, y: batch_ys})
			losses.append(batch_loss)
		if epoch % 10 == 0 and epoch > 0:
			print "Finished epoch %d, loss = %f" %(epoch, sum(losses) / len(losses))
			losses = []

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
	result = sess.run(loss, feed_dict=feed_dict)
	print "Test loss: %f" % result

if __name__ == '__main__':
	data_train, labels_train, data_dev, labels_dev, data_test, labels_test = loadJSON(input_file, [0.8, 0.8])
	params = train(data_train, labels_train)
	test(data_test, labels_test, params)


