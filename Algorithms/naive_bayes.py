import numpy as np
from util import loadTxtAsBuckets as loadTxt, loadJSONAsBuckets as loadJSON
from matplotlib import pyplot as plt

train_test_split = 0.7
result_path_prefix = "../results/naive_bayes/"
input_file = "../dataProcessing/Processed Recipes/_Cookies.json"
# input_file = "../data/train_data.txt"

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

	diff = np.zeros(posts.shape)
	for r in range(len(predictions)):
		diff[r][predictions[r]] = float('-inf')
	posts += diff
	second_guesses = np.argmax(posts, axis=1)

	return predictions, second_guesses

def getModel(num_categories, split, get_names=False):
	bucket_width = 4.0 / num_categories
	buckets = [1 + bucket_width * (i + 1) for i in range(num_categories - 1)]
	buckets.append(5)
	if get_names:
		train_inputs, train_labels, ingredient_list = loadJSON(input_file, split, buckets=buckets, get_names=True)
		model = train(train_inputs, train_labels, num_categories)
		return model, ingredient_list, buckets, bucket_width

	train_inputs, train_labels, test_inputs, test_labels = loadJSON(input_file, split, buckets=buckets)
	model = train(train_inputs, train_labels, num_categories)
	return model, test_inputs, test_labels

def evaluate(output, label):
	return (output != label).sum() * 1. / len(output)

def evalLenient(output, second_guesses, label):
	return max(((output != label).sum() + (second_guesses != label).sum() - len(output)) * 1. / len(output), 0)

def makePlot(xs, ys, title, filename):
	plt.scatter(xs, ys)
	plt.title(title)
	plt.savefig(filename + '.svg')
	plt.figure()

def makeTwoLinePlot(xs, y1s, y2s, title, filename):
	plt.scatter(xs, y1s)
	plt.scatter(xs, y2s)
	plt.title(title)
	plt.xlabel("Number of buckets")
	plt.ylabel("Error rate")
	plt.savefig(filename + '.svg')
	plt.figure()

def makeErrorPlots():
	breakups = range(1,9)
	splits = [0.5, 0.6, 0.7, 0.8, 0.9]
	for train_test_split in splits:
		errors = []
		lenient_errors = []
		for num_categories in breakups:
			model, test_inputs, test_labels = getModel(num_categories, train_test_split)

			predictions, second_guesses = test(test_inputs, model, num_categories)
			error = evaluate(predictions, test_labels)
			lenient_error = evalLenient(predictions, second_guesses, test_labels)

			errors.append(error)
			lenient_errors.append(lenient_error)
		else:
			makeTwoLinePlot(breakups, errors, lenient_errors, "1st/2nd guess accuracy", result_path_prefix + "nb_error_lenient_%dsplit" %int(10 * train_test_split))

def getIngredientScores(num_categories):
	model, ingredient_list, buckets, bucket_width = getModel(num_categories, 0.8, get_names=True)
	dists = model["phis"]
	dists /= np.reshape(np.sum(dists, axis=1), (dists.shape[0], 1))
	scores = np.dot(dists, np.transpose(np.array(buckets) - bucket_width / 2))

	ingredient_scores = [(ingredient_list[idx], scores[idx]) for idx in range(len(ingredient_list))]
	
	def higherScore(pair1, pair2):
		diff = pair1[1] - pair2[1]
		if diff == 0.0:
			return 1
		return int(diff / np.abs(diff))
		
	ingredient_scores.sort(higherScore)
	return ingredient_scores

def getOutliers(scores, n):
	print "%d worst ingredients:" %n
	for pair in scores[:n]:
		print "%s (%.2f)" %(pair[0], pair[1])

	print "\n%d best ingredients:" %n
	for pair in scores[-n:]:
		print "%s (%.2f)" %(pair[0], pair[1])

def generateRecipe(num_categories, units=100, label=0):
	model, ingredient_list, _, __ = getModel(num_categories, 1, get_names=True)
	probs = model['phis'][:, label]
	# print probs.shape
	cumulative = np.zeros(probs.shape)
	cumulative[0] = probs[0]
	for ind in range(1, len(probs)):
		cumulative[ind] = cumulative[ind-1] + probs[ind]
	# print cumulative

	result = np.zeros(cumulative.shape)
	for unit in range(units):
		v = np.random.rand()
		given = np.searchsorted(cumulative, v)
		result[given] += 1

	return result, ingredient_list

	# for ind in range(units):

	# bucket_width = 4.0 / num_categories
	# buckets = [1 + bucket_width * (i + 1) for i in range(num_categories - 1)]
	# buckets.append(5)
	# if ".txt" in input_file:
	# 	train_inputs, train_labels = loadTxt(input_file, buckets)
	# elif ".json" in input_file:
	# 	train_inputs, train_labels, ingredient_list = loadJSON(input_file, 1, buckets=buckets, get_names=True)

def printRecipes(recipes, ingredient_list):
	for recipe in recipes:
		print "Suggested recipe:"
		for ind in range(recipe.shape[0]):
			if recipe[ind]:
				print "%d of %s" %(recipe[ind], ingredient_list[ind])
		print ""

if __name__ == "__main__":
	# makeErrorPlots()
	# scores = getIngredientScores(4)
	# getOutliers(scores, 5)
	recipe1, ingredient_list = generateRecipe(4, units=50, label=3)
	recipe2, ingredient_list = generateRecipe(4, units=50, label=0)
	printRecipes([recipe1, recipe2], ingredient_list)





