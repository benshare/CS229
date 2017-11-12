import numpy as np
import json
from matplotlib import pyplot as plt

ingredient_list = ["a", "b", 'c', 'd']
m = -1
n = len(ingredient_list)
num_categories = 4

def dictToFreqVec(d, header):
	fv = np.zeros((1, n))
	for i in range(n):
		ingredient = header[i]
		if ingredient in d:
			fv[:, i] = d[ingredient]
	return fv

def loadData(data_file, buckets=[2, 3, 4, 5]):
	f = open(data_file, 'r')
	m = int(f.readline())
	# if num_categories == -1:
	# 	num_categories = len(buckets)

	inputs = np.zeros((m, n))
	labels = np.zeros((m,))

	for r in range(m):
		cur = f.readline()
		d = json.loads(cur)
		v = dictToFreqVec(d, ingredient_list)
		inputs[r, :] = v

		cur = f.readline()
		l = float(cur)
		b = 0
		while buckets[b] < l:
			b += 1
		labels[r] = b

	f.close()
	return inputs, labels

def train(matrix, category):
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

def test(matrix, state):
    m, n = matrix.shape
    phis = state["phis"]
    priors = state["priors"]
    posts = np.dot(matrix, np.log(phis)) + np.log(priors)
    predictions = np.argmax(posts, axis=1)

    return predictions

def evaluate(output, label):
    return (output != label).sum() * 1. / len(output)

if __name__ == "__main__":
	train_inputs, train_labels = loadData("train_data.txt")
	test_inputs, test_labels = loadData("test_data.txt")

	model = train(train_inputs, train_labels)
	predictions = test(test_inputs, model)

	print "Error: ", evaluate(predictions, test_labels)



