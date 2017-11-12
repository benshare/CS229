import numpy as np
from util import dictToFreqVec as dictToFreqVec, loadDataInBuckets as loadData

# Trains a naive Bayes model on the given input data (matrix)
# and labels (category). num_categories is needed to know how
# to bucket.
def train(matrix, category, num_categories):
    m, n = matrix.shape
    state = {}
    priors = [(np.sum(category == i) + 1.0) / num_categories for i in range(num_categories)]
    state["priors"] = np.array(priors)

    label_matrix = np.zeros((m, num_categories))
    for cat in range(num_categories):
    	newcol = category == cat
    	label_matrix[:, cat] = newcol

    phis = np.ones((n, num_categories))
    totals = np.zeros((num_categories))
    totals.fill(n)

    phis += np.dot(np.transpose(matrix), label_matrix)
    phis /= sum(phis)
    state["phis"] = phis
    return state

def test(matrix, state, num_categories):
    m, n = matrix.shape
    phis = state["phis"]
    priors = state["priors"]
    posts = np.dot(matrix, np.log(phis)) + np.log(priors)
    predictions = np.argmax(posts, axis=1)

    return predictions

def evaluate(output, label):
    return (output != label).sum() * 1. / len(output)

if __name__ == "__main__":
	num_categories = 4
	train_inputs, train_labels = loadData("../data/train_data.txt")
	test_inputs, test_labels = loadData("../data/test_data.txt")

	model = train(train_inputs, train_labels, num_categories)
	predictions = test(test_inputs, model, num_categories)

	print "Error: ", evaluate(predictions, test_labels)


