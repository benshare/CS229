import numpy as np
import json
from matplotlib import pyplot as plt

# Takes a dictionary (ingredient: freq) and a header
# (ordered listing of ingredients) and returns a
# corresponding 'one-hot' vector whose columns
# correspond to the frequency of the given header.
def dictToFreqVec(d, header):
	n = len(header)
	freq_vec = np.zeros((1, n))
	for i in range(n):
		ingredient = header[i]
		if ingredient in d:
			freq_vec[:, i] = d[ingredient]
	return freq_vec

def loadData(data_file, buckets=[2, 3, 4, 5]):
	f = open(data_file, 'r')
	m = int(f.readline())

	recipe_list = []
	label_list = []
	for recipe in range(m):
		cur = f.readline()
		as_dict = json.loads(cur)
		recipe_list.append(as_dict)

		cur = f.readline()
		label_list.append(float(cur))

	all_ingredients = set()
	for recipe in recipe_list:
		all_ingredients.update(recipe)
	ingredient_list = list(all_ingredients)
	n = len(ingredient_list)

	inputs = np.zeros((m, n))
	labels = np.zeros((m,))

	for recipe in range(m):
		as_freq_vec = dictToFreqVec(recipe_list[recipe], ingredient_list)
		inputs[recipe, :] = as_freq_vec

		cur_label = label_list[recipe]
		bucket_ind = 0
		while buckets[bucket_ind] < cur_label:
			bucket_ind += 1
		labels[recipe] = bucket_ind

	f.close()
	return inputs, labels

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


