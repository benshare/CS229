import numpy as np
import json

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

def bucketer(original, buckets):
	bucket_ind = 0
	while buckets[bucket_ind] < original:
		bucket_ind += 1
	return bucket_ind

def loadDataInBuckets(data_file, buckets=[2, 3, 4, 5]):
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
		print recipe
		all_ingredients.update(recipe)
	ingredient_list = list(all_ingredients)
	n = len(ingredient_list)

	inputs = np.zeros((m, n))
	labels = np.zeros((m,))

	for recipe in range(m):
		as_freq_vec = dictToFreqVec(recipe_list[recipe], ingredient_list)
		inputs[recipe, :] = as_freq_vec

		cur_label = label_list[recipe]
		labels[recipe] = bucketer(cur_label, buckets)

	f.close()
	return inputs, labels

def loadDataAsTokens(data_file):
	f = open(data_file, 'r')
	m = int(f.readline())

	recipe_list = []
	for recipe in range(m):
		cur = f.readline()
		as_dict = json.loads(cur)
		recipe_list.append(as_dict)

		cur = f.readline()

	all_ingredients = set()
	for recipe in recipe_list:
		all_ingredients.update(recipe)
	ingredient_list = list(all_ingredients)
	n = len(ingredient_list)

	inputs = np.zeros((m, n))

	for recipe in range(m):
		as_freq_vec = dictToFreqVec(recipe_list[recipe], ingredient_list)
		inputs[recipe, :] = as_freq_vec

	f.close()
	return inputs, ingredient_list

def loadDataFromJSON(file_name, split, buckets=[2, 3, 4, 5]):
	f = open("../data/_Brownies.json", 'r')
	obj = json.load(f)
	recipe_list = obj["_ingredients"]
	recipe_list = [{ing[1]: ing[0] for ing in recipe} for recipe in recipe_list]
	m = len(recipe_list)

	all_ingredients = set()
	for recipe in recipe_list:
		all_ingredients.update(recipe)
	ingredient_list = list(all_ingredients)
	n = len(ingredient_list)
	
	label_list = obj["rating"]
	label_list = [bucketer(l, buckets) for l in label_list]

	cutoff = int(m * split)

	input_train = np.zeros((cutoff, n))
	labels_train = np.zeros((cutoff,))
	input_test = np.zeros((m - cutoff, n))
	labels_test = np.zeros((m - cutoff,))

	for recipe in range(m):
		as_freq_vec = dictToFreqVec(recipe_list[recipe], ingredient_list)
		if recipe < cutoff:
			input_train[recipe, :] = as_freq_vec
			labels_train[recipe] = label_list[recipe]
		else:
			input_test[recipe - cutoff, :] = as_freq_vec
			labels_test[recipe - cutoff] = label_list[recipe]

	f.close()
	return input_train, labels_train, input_test, labels_test
