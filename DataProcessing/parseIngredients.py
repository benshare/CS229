import json  # Reading data
import nltk  # Stemming plurals

class IngredientParser():

    def __init__(self, data):
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
            'ounce': 1 * weightToVol,
            'pound': 16 * weightToVol,
            'gram': 0.035274 * weightToVol,
        }
        self.units = self.unitConversions.keys()

        self.lemma = nltk.wordnet.WordNetLemmatizer()

        # self.foods = ['sugar', 'shortening', 'eggs', 'bananas', 'caramel sauce', 'fudge sauce', 'milk', 'vanilla', 'flour', 'baking soda', 'baking powder', 'cocoa powder', 'cinnamon', 'nutmeg', 'walnuts', 'salt', 'water', 'yogurt', 'butter', 'egg', 'chocolate chips', 'oil', 'milk', 'brownie mix', 'yeast', 'raisins', 'cloves', 'banana', 'oats', 'zest', 'pecans', 'juice', 'squares', 'flowers', 'applesauce', 'seeds', 'mayonnaise', 'margarine', 'sour cream', 'honey', 'potatoes', 'ginger', 'wheat germ', 'tofu', 'meal', 'allspice', 'chocolate chunks', 'dates', 'vinegar', 'coconut', 'starch', 'cardamom', 'beer', 'root', 'baking powder', 'walnuts', 'syrup', 'strawberries', 'blueberries', 'apple', 'papaya', 'quarters', 'vanilla', 'drained', 'diced', 'bran', 'gum', 'rum', 'toasted', 'flakes)', 'cherries', 'cubed', 'garnish', 'cereal', 'cranberries', 'separated', 'pieces', 'C)', 'nuts', 'cheese', 'tea', 'carrots', 'Delight®)', 'granules', 'seed', 'Splenda®)', 'Blend', 'zucchini', 'link)', 'thawed', 'molasses', 'puree', 'spice', 'Pam®)', 'Honey®)', 'drippings', 'bacon', 'cornmeal', 'lengthwise', 'whisked', 'Granulated)', 'Light)', 'Blend)', 'substitute', 'Arthur®)', 'Mixture)', 'peanuts', 'flavoring', 'Topping', 'pureed', 'Mill®)', 'Icing:', 'cognac', 'pumpkin', 'demerara', 'Oil', 'Extract', 'peeled', 'Applesauce', 'topping']
        self.foods = ['sugar', 'flour', 'eggs', 'chocolate', 'water', 'salt', 'butter', 'vanilla', 'oil', 'baking soda', 'baking powder']

    def parseFraction(self, token):
        tokens = token.split('/')
        if (len(tokens) > 1):
            return int(tokens[0]) / int(tokens[1])
        else:
            return int(tokens[0])

    # Handles cases: N food, N/N food, N N/N food, N (N unit) food
    def getQuantity(self, tokens):
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

    def getUnit(self, tokens):
        for token in tokens:
            matches = [u for u in self.units if u in token]
            if (len(matches) > 0):
                return matches[0]
        return ''

    def getUnitlessQuantity(self, quantity, unit):
        if unit in self.units:
            return quantity * self.unitConversions[unit]
        else:
            return quantity

    # Determine what food is present in an ingredient
    def getFood(self, tokens):
        for token in tokens:
            matches = [f for f in self.foods if self.lemma.lemmatize(f) in self.lemma.lemmatize(token)]
            if (len(matches) > 0):
                return matches[0]
        return ''

    # Parse single ingredient token
    def parseIngredient(self, ingredient):
        tokens = ingredient.split(' ')
        amount = self.getUnitlessQuantity(self.getQuantity(tokens), self.getUnit(tokens))
        food = self.getFood(tokens)
        if food == '' or amount  <= 0:
            return []
        else:
            return [
                self.getUnitlessQuantity(self.getQuantity(tokens), self.getUnit(tokens)),
                self.getFood(tokens)
            ]

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
    filename = 'Brownies.json'
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
            "_ingredients": data_out['_ingredients']}))
    jsonFile.write("\n")
    jsonFile.close()

    print('Done!')
