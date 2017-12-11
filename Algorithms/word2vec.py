import numpy as np
from util import loadTxtAsTokens as loadTxt, loadJSONAsTokens as loadJSON
import os.path
from random import randrange
from sklearn.manifold import TSNE as tsne
from matplotlib import pyplot as plt
from copy import deepcopy
import tensorflow as tf

step_size = 1
batch_size = 20
epsilon = 10**(-6)
activation_fn = lambda x: x#tf.nn.sigmoid
gamma = .01

result_path_prefix = "../results/embeddings/"
input_path_prefix = "../dataProcessing/Processed Recipes/"
# input_file = "_Cookies.json"
input_file = "_Cookies_CBOWembeddings_20_20epochs"

def batches(recipes, size):
	m, n = recipes.shape
	shuffled = recipes[np.random.permutation(m)]
	for batch in range(m/size):
		batch_data = shuffled[size * batch: size * (batch + 1), :]
		ordering = np.random.permutation(n)
		yield batch_data, ordering

def expandRecipes(recipes):
	m, n = recipes.shape
	shuffled = recipes[np.random.permutation(m)]
	for recipe_num in range(m):
		recipe = np.reshape(shuffled[recipe_num, :], (1, n))
		expanded = np.repeat(recipe, n, axis=0)
		labels = deepcopy(expanded.diagonal())
		np.fill_diagonal(expanded, 0)
		indices = np.eye(n)

		# yield expanded, indices, labels
		for ind in range(n):
			yield np.reshape(expanded[ind,:], (1, n)), np.reshape(indices[ind,:], (1, n)), labels[ind]

def trainCBOW(matrix, max_epochs=5, hidden_size=5):
	m, n = matrix.shape
	print "Num recipes = %d" %m

	# x1 = tf.placeholder(tf.float32, [n, n])
	# b1 = tf.Variable(tf.zeros([1, hidden_size]))

	# x2 = tf.placeholder(tf.float32, [n, n])
	# b2 = tf.Variable(tf.zeros([1, hidden_size]))

	# W1 = tf.Variable(tf.random_normal([n, hidden_size]))

	# h1 = activation_fn(tf.matmul(x1, W1) + b1)
	# h2 = activation_fn(tf.matmul(x2, W1) + b2)
	# b3 = tf.Variable(tf.zeros([1]))

	# y_hat = tf.reduce_sum(tf.mul(h1, h2), axis=1) + b3
	# y = tf.placeholder(tf.float32, [n])
	x1 = tf.placeholder(tf.float32, [1, n])
	b1 = tf.Variable(tf.zeros([1, hidden_size]))
	W1 = tf.Variable(tf.random_normal([n, hidden_size]))

	x2 = tf.placeholder(tf.float32, [1, n])
	b2 = tf.Variable(tf.zeros([1, hidden_size]))
	W2 = tf.Variable(tf.random_normal([n, hidden_size]))

	h1 = activation_fn(tf.matmul(x1, W1) + b1)
	h2 = activation_fn(tf.matmul(x2, W2) + b2)
	b3 = tf.Variable(tf.zeros([1]))

	y_hat = tf.reduce_sum(tf.mul(h1, h2), axis=1) + b3
	y = tf.placeholder(tf.float32, [])

	diff = tf.abs(y - y_hat)
	loss = diff + gamma * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W1))#tf.reduce_mean(tf.abs(y - y_hat))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	inputs = deepcopy(matrix)
	embeddings = np.zeros((1, hidden_size))
	for epoch in range(max_epochs):
		prev_embeddings = deepcopy(embeddings)
		epoch_loss = 0
		count = 1
		for context, index, label in expandRecipes(inputs):
			# print context[0,:]
			# print index
			# print label
			feed_dict = {x1: context, x2: index, y: label}
			_, error, embeddings, pred = sess.run([train_step, diff, W1, y_hat], feed_dict=feed_dict)
			# print pred
			# raise Exception("aah")
			# print count, np.linalg.norm(embeddings), np.linalg.norm(context)
			epoch_loss += error
			# print error
			if count % (100 * n) == 0:
				print "Finished example %d" % (count / n)
				# print np.linalg.norm(embeddings)
				print "error: ", error
			count += 1
		epoch_loss /= m * n

		if np.linalg.norm(embeddings - prev_embeddings) < epsilon:
			print "Converged on epoch %d" %epoch
			break
		print "Finished epoch %d, loss = %f" %(epoch, epoch_loss)

	return embeddings

def trainSkipgram(inputs):
	pass

def writeEmbeddingFiles(dims):
	if ".txt" in input_file:
		train_inputs, tokens = loadTxt(input_path_prefix + input_file)
	elif ".json" in input_file:
		train_inputs, tokens = loadJSON(input_path_prefix + input_file)

	for hidden_size in dims:
		for me in [1, 5, 20]:
			embeddings = trainCBOW(train_inputs, max_epochs=me, hidden_size=hidden_size)

			stem = input_file[:input_file.find('.')]
			np.savetxt(result_path_prefix + stem + "_CBOWembeddings_%d_%depochs" %(hidden_size, me), embeddings)

			fn = result_path_prefix + stem + "_tokens"
			if not os.path.isfile(fn):
				f = open(fn, 'w+')
				f.write(str(tokens))
				f.close()

def loadEmbeddingFiles(input_file):#, dimension):
	# stem = input_file[:input_file.find('.')]
	f = open(result_path_prefix + input_file)#stem + "_embeddings_" + str(dimension))
	embeddings = np.loadtxt(f)
	f.close()

	f = open(result_path_prefix + "_Cookies" + "_tokens")
	tokens = eval(f.readline())
	f.close()
	return embeddings, tokens

def getKNeighbors(matrix, chosen_index, k, best=True):
	given = matrix[chosen_index][:]
	other_vecs = np.zeros((matrix.shape[0]-1, matrix.shape[1]))
	other_vecs[:chosen_index][:] = matrix[:chosen_index][:]
	other_vecs[chosen_index:][:] = matrix[chosen_index + 1:][:]

	distance_vectors = other_vecs - given
	distances = np.linalg.norm(distance_vectors, axis=1)

	if best:
		indices = np.argpartition(distances, k-1)[:k]
	else:
		indices = np.argpartition(distances, -k)[-k:]
	for i in range(len(indices)):
		if indices[i] >= chosen_index:
			indices[i] += 1
	return indices

def getKBestSubstitutes(ingredient, embeddings, tokens, k):
	chosen_index = tokens.index(ingredient)
	found = getKNeighbors(embeddings, chosen_index, k)
	return [tokens[idx] for idx in found]

def getKWorstSubstitutes(ingredient, embeddings, tokens, k):
	chosen_index = tokens.index(ingredient)
	found = getKNeighbors(embeddings, chosen_index, k, best=False)
	return [tokens[idx] for idx in found]

def getTestEmbeddings(m, d):
	embeddings = np.zeros((m, d))
	for i in range(m / 4):
		base = np.array([randrange(10) for _ in range(d)])
		diff = np.zeros((1, d))
		diff[:][0] = 1
		embeddings[i*4][:] = base
		embeddings[i*4+1][:] = base + diff
		embeddings[i*4+2][:] = base + diff * 3
		embeddings[i*4+3][:] = base + diff * 6
	return embeddings, [str(i) for i in range(m)]

def project(embeddings, tokens):
	print "Running tsne"

	projected = tsne().fit_transform(embeddings)

	plt.figure()
	plt.scatter(projected[:, 0], projected[:, 1])
	for label, x, y in zip(tokens, projected[:, 0], projected[:, 1]):
		plt.annotate(label, xy=(x, y), xytext=(-10, -10), textcoords='offset points', size='x-small')
	plt.savefig("../results/embeddings/projection")

def highlight(embeddings, tokens, keyword):
	print "Running tsne"

	projected = tsne().fit_transform(embeddings)
	colors = np.array([keyword in t for t in tokens])

	plt.figure()
	plt.scatter(projected[:, 0], projected[:, 1], c=colors)
	for label, x, y in zip(tokens, projected[:, 0], projected[:, 1]):
		if keyword in label:
			plt.annotate(label, xy=(x, y), xytext=(-10, -10), textcoords='offset points', size='x-small')
	plt.savefig("../results/embeddings/highlight_\"%s\"" %keyword)

def getKNeighbors(matrix, chosen_index, k, best=True):
	given = matrix[chosen_index][:]
	other_vecs = np.zeros((matrix.shape[0]-1, matrix.shape[1]))
	other_vecs[:chosen_index][:] = matrix[:chosen_index][:]
	other_vecs[chosen_index:][:] = matrix[chosen_index + 1:][:]

	distance_vectors = other_vecs - given
	distances = np.linalg.norm(distance_vectors, axis=1)

	if best:
		indices = np.argpartition(distances, k-1)[:k]
	else:
		indices = np.argpartition(distances, -k)[-k:]
	for i in range(len(indices)):
		if indices[i] >= chosen_index:
			indices[i] += 1
	return indices

def getKBestSubstitutes(ingredient, embeddings, tokens, k):
	chosen_index = tokens.index(ingredient)
	found = getKNeighbors(embeddings, chosen_index, k)
	return [tokens[idx] for idx in found]

def getKWorstSubstitutes(ingredient, embeddings, tokens, k):
	chosen_index = tokens.index(ingredient)
	found = getKNeighbors(embeddings, chosen_index, k, best=False)
	return [tokens[idx] for idx in found]

def evaluate(ingredient, embeddings, tokens, k=5):
	best = getKBestSubstitutes(ingredient, embeddings, tokens, k)
	worst = getKWorstSubstitutes(ingredient, embeddings, tokens, k)

	f = open(result_path_prefix + 'substitutes_\"%s\"' % ingredient, 'w+')
	f.write("Best substitutes:\n")
	for i in best:
		f.write(i + '\n')

	f.write("\nWorst substitutes:\n")
	for i in worst:
		f.write(i + '\n')

	f.close()

if __name__ == "__main__":
	# writeEmbeddingFiles([2, 5, 20])

	embeddings, tokens = loadEmbeddingFiles(input_file)#, hidden_size)
	# targets = ["semi-sweet chocolate"]
	# project(embeddings, tokens)
	# highlight(embeddings, tokens, "peanut butter")
	evaluate("milk", embeddings, tokens)


	# m = 12
	# d = 10
	# embeddings, tokens = getTestEmbeddings(m, d)
	# for idx in range(m):
	# 	print "neighbor for %d:" %idx
	# 	print getKSubstituteSuggestions(str(idx), embeddings, tokens, 1)

