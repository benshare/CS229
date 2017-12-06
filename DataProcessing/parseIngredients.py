import json  # Reading data
import string
import cleanIngredients

class IngredientParser():

    def __init__(self, data, cleanFoodFilename, debug=False):
        self.debug = debug
        self.data = data
        self.recipeIngredients = data['ingredients']

        # volume: fluid ounces / unit
        # weight: ounces / unit
        weightToVol = 4./3  # approximation of 1/density for all foods; 4 fluid ounces = 3 ounces
        self.unitConversions = {
            'teaspoon': 1./6,
            'tablespoon': 1./2,
            'cup': 8,
            'pint': 16,
            'quart': 32,
            'liter': 34,
            'gallon': 128,

            'pinch': 1./8,  # from rigorous measurements
            'dash': 1./4,
            'drop': 0.0016907,
            'packet': 1./3,  # artifical sweetener
            'clove': 2./3,  # garlic
            'square': 1 * weightToVol,  # chocolate
            'bar': 4.4 * weightToVol,  # chocolate

            'ounce': 1 * weightToVol,
            'pound': 16 * weightToVol,
            'gram': 0.035274 * weightToVol,
        }

        self.singleFoodVolumes = { # to make "1 onion" and "1 cup chopped onion" comparable
            'egg': 3./2,
            'banana': 20,
            'onion': 8,
            'zucchini': 8,
            'eggplant': 24,
            'tomato': 4,
            'thyme': 1./4,
            'oregano': 1./4
        }
        self.units = self.unitConversions.keys()

        with open(cleanFoodFilename, 'r') as cleanNamesFile:
            self.cleanFoodNames = json.load(cleanNamesFile)

    def normalizeStrings(self, strings):
        exclude = set(string.punctuation+' ')
        out = []
        for s in strings:
            out.append(''.join(ch for ch in s if ch not in exclude).lower())
        return out

    def parseFraction(self, token):
        tokens = token.split('/')
        if (len(tokens) > 1):
            return int(tokens[0]) / int(tokens[1])
        else:
            return int(tokens[0])

    # Handles cases: N food, N/N food, N N/N food, N (N unit) food
    def getQuantity(self, ingredient):
        tokens = ingredient.split(' ')
        n = -1
        try:
            n = self.parseFraction(tokens[0])
            if tokens[1][0] == '(':
                n *= float(tokens[1][1:])
            else:
                try:
                    n += self.parseFraction(tokens[1])
                except:
                    return n
        except:
            return n
        return n

    def getUnit(self, ingredient):
        for unit in self.units:
            if unit in ingredient:
                return unit
        return ''

    def getUnitlessQuantity(self, ingredient, food, quantity, unit):
        if unit in self.units:
            return quantity * self.unitConversions[unit]
        elif food in self.singleFoodVolumes.keys():
            return quantity * self.singleFoodVolumes[food]
        else:
            # if self.debug:
            #     print("Unable to find unit in: " + ingredient)
            return quantity

    # Determine what food is present in an ingredient
    def getFood(self, ingredient):
        return cleanIngredients.match_ingredient(ingredient, self.cleanFoodNames)

    # Parse single ingredient token
    def parseIngredient(self, ingredient):
        food = self.getFood(ingredient)
        amount = self.getUnitlessQuantity(ingredient, food, self.getQuantity(ingredient), self.getUnit(ingredient))

        if self.debug:
            print(ingredient,"-",[amount,food])

        if food == '' or amount  <= 0:
            return []
        else:
            return [amount, food]

    # Process all recipes and ingredients passed from initialized
    # foo = input, _foo = output
    def parseAll(self):
        _recipes = []
        count = 0
        for i in range(len(self.recipeIngredients)):
            recipe = self.recipeIngredients[i]

            _recipe = []
            for ingredient in recipe:
                _ingredient = self.parseIngredient(ingredient)
                if not(_ingredient == []):
                    _recipe.append(_ingredient)
            _recipes.append(_recipe)

            count += 1
            if count % 50 == 0:
                print (count, 'done')

        self.data['_ingredients'] = _recipes
        return self.data;


if __name__ == "__main__":
    # Read data
    data = None
    filename = 'test.json'
    with open(filename) as data_file:
        data = json.load(data_file)

    # Parse ingredients
    data_out = IngredientParser(data).parseAll()

    # Save data
    jsonFile = open('_' + filename, "w")
    jsonFile.truncate()
    jsonFile.write(json.dumps({
            "name": data_out['name'],
            "rating": data_out['rating'],
            "reviews": data_out['reviews'],
            "servings": data_out['servings'],
            "_ingredients": data_out['_ingredients']}))
    jsonFile.write("\n")
    jsonFile.close()

    print('Done!')
