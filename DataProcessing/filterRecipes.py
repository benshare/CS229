import json  # Reading data

class RecipeFilterer():

    def __init__(self, data):
        self.data = data
        self.min_reviews = 3

    def filterAll(self):
        n_recipes = len(self.data['name'])
        good_indices = []

        for i in range(n_recipes):
            valid = True

            if self.data['reviews'][i] < self.min_reviews:
                valid = False

            if self.data['rating'][i] < 1:  # Lowest possible rating is one star
                valid = False

            if len(self.data['_ingredients'][i]) == 0:
                valid = False

            if valid:
                good_indices.append(i)

        data_out = {
            'name' : [i for j, i in enumerate(self.data['name']) if j in good_indices],
            'rating' : [i for j, i in enumerate(self.data['rating']) if j in good_indices],
            'reviews' : [i for j, i in enumerate(self.data['reviews']) if j in good_indices],
            '_ingredients' : [i for j, i in enumerate(self.data['_ingredients']) if j in good_indices]
        }
        return data_out

if __name__ == "__main__":
    # Read data
    data = None
    filename = '_test.json'
    with open(filename) as data_file:
        data = json.load(data_file)

    # Parse ingredients
    data_out = RecipeFilterer(data).filterAll()

    # Save data
    jsonFile = open('_' + filename, "w")
    jsonFile.truncate()
    jsonFile.write(json.dumps({
            "name": data_out['name'],
            "rating": data_out['rating'],
            "reviews": data_out['reviews'],
            "_ingredients": data_out['_ingredients']}))
    jsonFile.write("\n")
    jsonFile.close()

    print('Done!')