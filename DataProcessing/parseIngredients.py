import json  # Reading data
import nltk  # Stemming plurals

#TODO deal with 1 (xxx ounce) format, weight units

class IngredientParser():

    def __init__(self, ingredients):
        self.ingredients = ingredients
        self.ingredients_ = []
        self.unitConversions = {
            'teaspoon': 1./6,
            'tablespoon': 1./2,
            'cup': 8,
            'pint': 16,
            'quart': 32,
            'liter': 34,
            'gallon': 128,
        }
        self.units = self.unitConversions.keys()

        self.lemma = nltk.wordnet.WordNetLemmatizer()

        # self.foods = ['sugar', 'shortening', 'eggs', 'bananas', 'caramel sauce', 'fudge sauce', 'milk', 'vanilla', 'flour', 'baking soda', 'baking powder', 'cocoa powder', 'cinnamon', 'nutmeg', 'walnuts', 'salt', 'water', 'yogurt', 'butter', 'egg', 'chocolate chips', 'oil', 'milk', 'brownie mix', 'yeast', 'raisins', 'cloves', 'banana', 'oats', 'zest', 'pecans', 'juice', 'squares', 'flowers', 'applesauce', 'seeds', 'mayonnaise', 'margarine', 'sour cream', 'honey', 'potatoes', 'ginger', 'wheat germ', 'tofu', 'meal', 'allspice', 'chocolate chunks', 'dates', 'vinegar', 'coconut', 'starch', 'cardamom', 'beer', 'root', 'baking powder', 'walnuts', 'syrup', 'strawberries', 'blueberries', 'apple', 'papaya', 'quarters', 'vanilla', 'drained', 'diced', 'bran', 'gum', 'rum', 'toasted', 'flakes)', 'cherries', 'cubed', 'garnish', 'cereal', 'cranberries', 'separated', 'pieces', 'C)', 'nuts', 'cheese', 'tea', 'carrots', 'Delight®)', 'granules', 'seed', 'Splenda®)', 'Blend', 'zucchini', 'link)', 'thawed', 'molasses', 'puree', 'spice', 'Pam®)', 'Honey®)', 'drippings', 'bacon', 'cornmeal', 'lengthwise', 'whisked', 'Granulated)', 'Light)', 'Blend)', 'substitute', 'Arthur®)', 'Mixture)', 'peanuts', 'flavoring', 'Topping', 'pureed', 'Mill®)', 'Icing:', 'cognac', 'pumpkin', 'demerara', 'Oil', 'Extract', 'peeled', 'Applesauce', 'topping']

    def parseFraction(self, token):
        tokens = token.split('/')
        if (len(tokens) > 1):
            return int(tokens[0]) / int(tokens[1])
        else:
            return int(tokens[0])

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

    # Assumes tokens is lemmatized
    def getUnit(self, tokens):
        for token in tokens:
            if token in self.units:
                return token
        return ''

    def parse(self):
        for ingredient in self.ingredients:
            tokens = ingredient.split(' ')
            tokens_ = [self.lemma.lemmatize(token) for token in tokens]
            print(self.getQuantity(tokens), self.getUnit(tokens_))

        # for ingredient in self.ingredients:
        #     tokens = ingredient.split(' ')
        #     if not(tokens[-1] in self.ingredients_):
        #         self.ingredients_.append(tokens[-1])
        # print(self.ingredients_)


def main():
    data = None
    with open('BananaBread.json') as data_file:
        data = json.load(data_file)

    recipe = 0
    ingredients = data['ingredients'][recipe]
    # print('Ingredients before:', ingredients)
    # result = IngredientParser(ingredients).parse()
    # print('Ingredients after:', result)

    ingredients_ = []
    for ingredients in data['ingredients']:
        for ingredient in ingredients:
            tokens = ingredient.split(' ')
            if not(tokens[-1] in ingredients_):
                ingredients_.append(tokens[-1])
    print(ingredients_)


main()
