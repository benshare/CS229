import numpy as np
from util import dictToFreqVec as dictToFreqVec, loadDataAsTokens as loadData

dim = 4
max_iters = 30
step_size = 1
zero_center = True
path_prefix = "../results/vectors/"

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

	return in_embeddings

if __name__ == "__main__":
	train_inputs, tokens = loadData("../data/w2v_data.txt")

	embeddings = train(train_inputs)
	np.savetxt(path_prefix + "embeddings_%d" %dim, embeddings)

	f = open(path_prefix + "tokens_%d" %dim, 'w+')
	f.write(str(tokens))
	f.close()

