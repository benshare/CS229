import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

def main():
    with open('_Brownies.json') as data_file:
        data = json.load(data_file)

        plt.figure()
        plt.title("Number of Stars")
        plt.plot(sorted(data['rating']))
        plt.show()

        plt.figure()
        plt.title("Number of Reviews")
        plt.plot(sorted(data['reviews']))
        plt.yscale('log')
        plt.show()

        plt.figure()
        a = np.array([data['rating'],data['reviews']]).T
        a = a[a[:, 0].argsort()]
        print(a)
        plt.plot(a[:,0]*1000)
        plt.plot(a[:,1])
        plt.show()  # Blue = # stars * 1000, Orange = # reviews

        sorted_ingredients = []
        sorted_ingredient_counts = []
        flat_ingredients = [item[1] for sublist in data['_ingredients'] for item in sublist]
        ingredient_occurences = Counter(flat_ingredients)
        for w in sorted(ingredient_occurences, key=ingredient_occurences.__getitem__):
            sorted_ingredients.append(w)
            sorted_ingredient_counts.append(ingredient_occurences[w])

        print(ingredient_occurences)

        plt.figure()
        plt.title("Ingredient frequencies")
        x = list(range(len(sorted_ingredients)))
        plt.bar(x[-10:], sorted_ingredient_counts[-10:])
        plt.xticks(x[-10:], sorted_ingredients[-10:])
        plt.show()

main()
