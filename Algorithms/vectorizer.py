import numpy as np
from util import loadTxtAsTokens as loadTxt, loadJSONAsTokens as loadJSON
import os.path
from random import randrange
from sklearn.manifold import TSNE as tsne
from matplotlib import pyplot as plt
from copy import deepcopy

dim = 2
max_iters = 500
step_size = .1
zero_center = True
dimension_limit = 100
epsilon = 10**(-8)

result_path_prefix = "../results/embeddings/"
input_path_prefix = "../dataProcessing/Processed Recipes/"
input_file = "_Cookies.json"

def train(matrix):
	m, n = matrix.shape
	cooccurences = np.dot(np.transpose(matrix), matrix)
	for i in range(n):
		cooccurences[i][i] = 0
	cooccurences = np.transpose(cooccurences / np.sum(cooccurences, axis=1))

	in_embeddings = np.random.normal(0, 1, (n, dim))
	out_embeddings = np.random.normal(0, 1, (n, dim))

	for iteration in range(max_iters):
		prev_embedding = deepcopy(in_embeddings)
		interval = 10
		if iteration % interval == 0:
			print "Iteration %d" %iteration
		in_update = np.dot(cooccurences, out_embeddings) * step_size
		out_update = np.dot(cooccurences, in_embeddings) * step_size
		in_embeddings += in_update
		out_embeddings += out_update

		if zero_center:
			in_embeddings -= np.mean(in_embeddings, axis=0)
			out_embeddings -= np.mean(out_embeddings, axis=0)

		in_embeddings /= (np.max(np.abs(in_embeddings), axis=0) / dimension_limit)
		out_embeddings /= (np.max(np.abs(out_embeddings), axis=0) / dimension_limit)

		cur_embedding = deepcopy(in_embeddings)
		diff = np.linalg.norm(prev_embedding - cur_embedding)
		if diff < epsilon:
			print "Converged on iteration %d" %iteration
			break

	return in_embeddings

def writeEmbeddingFiles():
	if ".txt" in input_file:
		train_inputs, tokens = loadTxt(input_path_prefix + input_file)
	elif ".json" in input_file:
		train_inputs, tokens = loadJSON(input_path_prefix + input_file)
	
	embeddings = train(train_inputs)

	stem = input_file[:input_file.find('.')]
	np.savetxt(result_path_prefix + stem + "_embeddings_%d" %dim, embeddings)

	fn = result_path_prefix + stem + "_tokens"
	if not os.path.isfile(fn):
		f = open(fn, 'w+')
		f.write(str(tokens))
		f.close()

def loadEmbeddingFiles(input_file, dimension):
	stem = input_file[:input_file.find('.')]
	f = open(result_path_prefix + stem + "_embeddings_" + str(dimension))
	embeddings = np.loadtxt(f)
	f.close()

	f = open(result_path_prefix + stem + "_tokens")
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

def project(embeddings, tokens, selectedTokens):
	print "Running tsne"

	projected = tsne().fit_transform(embeddings)
	colors = np.array([t in selectedTokens for t in tokens])

	plt.figure()
	plt.scatter(projected[:, 0], projected[:, 1], c=colors)
	plt.savefig("../results/embeddings/projection")

if __name__ == "__main__":
	writeEmbeddingFiles()

	embeddings, tokens = loadEmbeddingFiles(input_file, dim)
	best = getKBestSubstitutes("milk chocolate", embeddings, tokens, 7)
	worst = getKWorstSubstitutes("milk chocolate", embeddings, tokens, 7)
	print "Best substitutes:", best
	print "Worst substitutes:", worst
	project(embeddings, tokens, best)

