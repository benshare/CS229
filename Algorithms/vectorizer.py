import numpy as np
from util import loadTxtAsTokens as loadTxt, loadJSONAsTokens as loadJSON
import os.path

dim = 100
max_iters = 1000
step_size = 1
zero_center = True
dimension_limit = 100

result_path_prefix = "../results/embeddings/"
input_path_prefix = "../data/"
input_file = "_Brownies.json"

def train(matrix):
	m, n = matrix.shape
	cooccurences = np.dot(np.transpose(matrix), matrix)
	for i in range(n):
		cooccurences[i][i] = 0
	cooccurences = np.transpose(cooccurences / np.sum(cooccurences, axis=1))

	in_embeddings = np.random.normal(0, 1, (n, dim))
	out_embeddings = np.random.normal(0, 1, (n, dim))

	for iteration in range(max_iters):
		if iteration % 10 == 0:
			print "Iteration %d" %iteration
		in_update = np.dot(cooccurences, out_embeddings) / step_size
		out_embeddings = np.dot(cooccurences, in_embeddings) / step_size
		in_embeddings += in_update
		out_embeddings += out_embeddings

		if zero_center:
			in_embeddings -= np.mean(in_embeddings, axis=0)
			out_embeddings -= np.mean(out_embeddings, axis=0)

		in_embeddings /= (np.max(np.abs(in_embeddings), axis=0) / dimension_limit)
		out_embeddings /= (np.max(np.abs(out_embeddings), axis=0) / dimension_limit)

	return in_embeddings

if __name__ == "__main__":
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

