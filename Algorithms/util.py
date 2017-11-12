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