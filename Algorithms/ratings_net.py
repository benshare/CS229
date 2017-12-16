import tensorflow as tf
import numpy as np
from scipy.special import erfc, erfcinv
from util import loadJSONForNet as loadJSON
from matplotlib import pyplot as plt

FLAGS = None

hidden_size = 20
dish = "Cookies"
input_file = "../dataProcessing/Processed Recipes/_" + dish + ".json"
activation_fn = tf.nn.sigmoid
max_epochs = 100
batch_size = 20
gamma = 10**(-4)

def undoLabelNormalization(all_data):
	index_list = [1, 5]
	label_list = [all_data[ind] for ind in index_list]
	mus = []
	sigmas = []
	for i in range(len(index_list)):
		labels = label_list[i]
		mu = np.mean(labels)
		sigma = np.std(labels)
		mus.append(mu)
		sigmas.append(sigma)

		modified_labels = (labels - mu) / sigma
		modified_labels = erfc(-modified_labels / 2)/2
		modified_labels = modified_labels * 4 + 1
		all_data[index_list[i]] = modified_labels
	return mus, sigmas

def renormalize(dists, mu, sigma):
	for i in range(len(dists)):
		dist = dists[i]
		normalized = (dist - 1) / 4
		normalized = -erfcinv(normalized * 2) * 2
		normalized = normalized * sigma + mu
		dists[i] = normalized

def batches(tensors, size, random=True):
	l = tensors[0].shape[0]
	if random:
		perm = np.random.permutation(l)
		tensors = [t[perm] for t in tensors]
	for batch in range(l/size):
		batch_data = tensors[0][size * batch: size * (batch + 1), :]
		batch_labels = tensors[1][size * batch: size * (batch + 1)]
		yield batch_data, batch_labels

def makeGraph(xs, ys, title, fn, setAxes=False, plotLine=True):
	plt.figure()
	plt.scatter(xs, ys, c='b')
	plt.title(title)
	if setAxes:
		plt.xlim(0.5, 5.5)
		plt.ylim(0.5, 5.5)
		if plotLine:
			plt.plot([1, 5], [1, 5], 'r')
		plt.xlabel("Ground truth")
		plt.ylabel("Model predictions")
	plt.savefig("../results/ratings_net/" + dish + '/' + fn)

def bucket(before, num_buckets = 4):
	width = (4.0) / num_buckets
	cutoffs = [2 + width * i for i in range(num_buckets - 1)]
	after = [np.searchsorted(cutoffs, b)[0] for b in before]
	return after

def makePredOneHot(before, penalize=False):
	after = np.zeros(before.shape)
	return np.eye(before.shape[1])[np.array(np.argmax(before, axis=1)).reshape(-1)]

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

	diff = tf.reduce_mean(tf.squared_difference(y, y_hat))
	norms = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
	loss = diff + gamma * norms
	train_step = tf.train.AdamOptimizer(0.02, 0.1).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Now, train
	epoch_losses = []
	epoch_norms = []
	epoch_params = []
	for epoch in range(1, max_epochs + 1):
		batch_losses = []
		batch_norms = []
		for batch_xs, batch_ys in batches([data, labels], batch_size):
			_, batch_loss, params, batch_norm = sess.run([train_step, diff, [W1, b1, W2, b2], norms], feed_dict={x: batch_xs, y: batch_ys})
			batch_losses.append(batch_loss)
			batch_norms.append(batch_norm)

		epoch_loss = sum(batch_losses) / len(batch_losses)
		epoch_losses.append(epoch_loss)
		epoch_norm = sum(batch_norms) / len(batch_norms)
		epoch_norms.append(epoch_norm)
		epoch_params.append(params)

		if epoch % 20 == 0 and epoch > 0:
			print "Finished epoch %d, loss = %f" %(epoch, epoch_loss)

	makeGraph(range(max_epochs), np.log(epoch_losses), "Training loss", "train_loss")
	makeGraph(range(max_epochs), epoch_norms, "Weight norms", "train_norms")

	return epoch_params

def trainBuckets(data, labels, num_buckets=4):
	M = data.shape[0]
	N = data.shape[1]

	# Create the model
	x = tf.placeholder(tf.float32, [None, N])
	W1 = tf.Variable(tf.random_normal([N, hidden_size]))
	b1 = tf.Variable(tf.zeros([1, hidden_size]))

	h = activation_fn(tf.matmul(x, W1) + b1)
	W2 = tf.Variable(tf.random_normal([hidden_size, num_buckets]))
	b2 = tf.Variable(tf.zeros(1))

	y_hat = tf.matmul(h, W2) + b2
	y = tf.placeholder(tf.int32, [None])

	diff = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
	norms = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
	loss = diff + gamma * norms
	train_step = tf.train.AdamOptimizer(0.02).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Now, train
	epoch_losses = []
	epoch_norms = []
	epoch_params = []
	for epoch in range(1, max_epochs + 1):
		batch_losses = []
		batch_norms = []
		for batch_xs, batch_ys in batches([data, labels], batch_size):
			feed_dict = {x: batch_xs, y: np.reshape(bucket(batch_ys), [batch_size])}
			_, batch_loss, params, batch_norm = sess.run([train_step, diff, [W1, b1, W2, b2], norms], feed_dict=feed_dict)
			batch_losses.append(batch_loss)
			batch_norms.append(batch_norm)

		epoch_loss = sum(batch_losses) / len(batch_losses)
		epoch_losses.append(epoch_loss)
		epoch_norm = sum(batch_norms) / len(batch_norms)
		epoch_norms.append(epoch_norm)
		epoch_params.append(params)

		if epoch % 20 == 0 and epoch > 0:
			print "Finished epoch %d, loss = %f" %(epoch, epoch_loss)

	makeGraph(range(max_epochs), np.log(epoch_losses), "Training loss", "train_loss")
	makeGraph(range(max_epochs), epoch_norms, "Weight norms", "train_norms")

	return epoch_params

def test(data, labels, params, mu=None, sigma=None, fn=""):
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
	makeGraph(labels, predictions, "Labels vs. predictions", (fn if fn else "test_preds"), setAxes=True)
	which = "Dev"
	if 'train' in fn:
		which = "Train"
	print "%s loss: %f" % (which, result)

def testRegressionToBuckets(data, labels, params, num_buckets=4):
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

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	feed_dict = {x: data, W1: params[0], b1: params[1], W2: params[2], b2: params[3]}
	predictions = sess.run([y_hat], feed_dict=feed_dict)[0]

	preds_bucketed = bucket(predictions, num_buckets)
	labels_bucketed = bucket(labels, num_buckets)
	mat = np.zeros([num_buckets, num_buckets])
	correct = 0.0
	for ind in range(M):
		x_val = labels_bucketed[ind]
		y_val = preds_bucketed[ind]
		mat[x_val, y_val] += 1
		if x_val == y_val:
			correct += 1

	print mat
	print "Accuracy: %f" % (100 * correct / M)

def testBuckets(data, labels_before, param_list, fn=None, num_buckets=4):
	M = data.shape[0]
	N = data.shape[1]
	labels = bucket(labels_before, num_buckets)

	# Create the model
	x = tf.placeholder(tf.float32, data.shape)
	W1 = tf.placeholder(tf.float32, [N, hidden_size])
	b1 = tf.placeholder(tf.float32, [1, hidden_size])

	h = activation_fn(tf.matmul(x, W1) + b1)
	W2 = tf.placeholder(tf.float32, [hidden_size, num_buckets])
	b2 = tf.placeholder(tf.float32, [1])

	y_hat = tf.matmul(h, W2) + b2

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	
	predictions_by_epoch = []
	accuracy_by_epoch = []
	for params in param_list:
		feed_dict = {x: data, W1: params[0], b1: params[1], W2: params[2], b2: params[3]}
		predictions = sess.run(y_hat, feed_dict=feed_dict)
		indices = np.argmax(predictions, axis=1)
		predictions_by_epoch.append(indices)
		accuracy_by_epoch.append(np.mean([(1 if indices[i] == labels[i] else 0) for i in range(M)]))

	best_epoch = np.argmax(accuracy_by_epoch)
	best_preds = predictions_by_epoch[best_epoch]

	mat = np.zeros([num_buckets, num_buckets])
	for ind in range(M):
		x_val = labels[ind]
		y_val = best_preds[ind]

		mat[x_val, y_val] += 1

	print "Best epoch: %d" %best_epoch
	print mat
	print "Accuracy: %f" % (100 * accuracy_by_epoch[best_epoch])

def test_bunch(data, labels, param_list, mu=None, sigma=None, fn=""):
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

	epoch_losses = []
	for params in param_list:
		feed_dict = {x: data, y: labels, W1: params[0], b1: params[1], W2: params[2], b2: params[3]}
		result = sess.run([loss], feed_dict=feed_dict)[0]
		epoch_losses.append(result)

	makeGraph(range(max_epochs), np.log(epoch_losses), "Testing loss", "test_loss")
	best_ind = np.argmin(epoch_losses)
	return best_ind, epoch_losses[best_ind]

if __name__ == '__main__':

	def doRegression():
		param_list = train(data_train, labels_train)

		best_epoch, best_loss = test_bunch(data_test, labels_test, param_list)
		print "Best epoch:", best_epoch
		best_params = param_list[best_epoch]

		test(data_dev, labels_dev, best_params)
		test(data_train, labels_train, best_params, fn="train_preds")

	def doClassification():
		param_list = trainBuckets(data_train, labels_train)
		testBuckets(data_test, labels_test, param_list)

	# See util.py for explanation of loading process.
	# [0.6, 0.8] means 0.6 train, 0.2 dev, 0.2 test.
	all_data = list(loadJSON(input_file, [0.6, 0.8]))

	# Optional; spreads data out to approximate a uniform distribution.
	# mus/sigmas are needed if wanting to renormalize (see fn above).
	mus, sigmas = undoLabelNormalization(all_data)

	data_train, labels_train, data_dev, labels_dev, data_test, labels_test = all_data
	print "Train size:", data_train.shape[0]
	print "Test size:", data_test.shape[0]

	doRegression()
	# doClassification()



