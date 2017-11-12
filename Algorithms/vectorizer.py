import numpy as np
from util import dictToFreqVec as dictToFreqVec, loadDataInBuckets as loadData


def train(matrix):
    m, n = matrix.shape
    embeddings = {}
    return embeddings

if __name__ == "__main__":
	num_categories = 4
	train_inputs = loadData("../data/train_data.txt")[0]

	model = train(train_inputs)
