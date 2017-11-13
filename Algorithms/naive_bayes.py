import numpy as np
from util import dictToFreqVec as dictToFreqVec, loadDataInBuckets as loadData, loadDataFromJSON as loadJSON
from matplotlib import pyplot as plt

train_test_split = 0.7
input_type = "JSON"
path_prefix = "../results/naive_bayes/"

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

def makePlot(xs, ys, title, filename):
	plt.scatter(xs, ys)
	plt.title(title)
	plt.savefig(filename)
	plt.figure()

if __name__ == "__main__":
	breakups = range(1,9)
	splits = [0.5, 0.6, 0.7, 0.8, 0.9]
	for train_test_split in splits:
		errors = []
		for num_categories in breakups:
			bucket_width = 4.0 / num_categories
			buckets = [1 + bucket_width * (i + 1) for i in range(num_categories - 1)]
			buckets.append(5)
			if input_type == "JSON":
				train_inputs, train_labels, test_inputs, test_labels = loadJSON("../data/_Brownies.json", train_test_split, buckets)
			elif input_type == "txt":
				train_inputs, train_labels = loadData("../data/train_data.txt", buckets)
				test_inputs, test_labels = loadData("../data/test_data.txt", buckets)

			model = train(train_inputs, train_labels, num_categories)
			predictions = test(test_inputs, model, num_categories)

			error = evaluate(predictions, test_labels)
			errors.append(error)
		makePlot(breakups, errors, "Error vs. bucket number", path_prefix + "nb_error_%dsplit" %int(10 * train_test_split))


