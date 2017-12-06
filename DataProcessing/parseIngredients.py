import json  # Reading data
import nltk  # Stemming plurals
import string

class IngredientParser():

    def __init__(self, data, debug=False):
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
            'ounce': 1 * weightToVol,
            'pound': 16 * weightToVol,
            'gram': 0.035274 * weightToVol,
        }

        self.singleFoodVolumes = { # to make "1 onion" and "1 cup chopped onion" comparable
            'onion': 8,
            'zucchini': 8,
            'eggplant': 24,
            'tomato': 4,
            'thyme': 1./4,
            'oregano': 1./4
        }
        self.units = self.unitConversions.keys()

        # TEST BROWNIES
        # self.foods = ['sugar', 'shortening', 'eggs', 'bananas', 'caramel sauce', 'fudge sauce', 'milk', 'vanilla', 'flour', 'baking soda', 'baking powder', 'cocoa powder', 'cinnamon', 'nutmeg', 'walnuts', 'salt', 'water', 'yogurt', 'butter', 'egg', 'chocolate chips', 'oil', 'milk', 'brownie mix', 'yeast', 'raisins', 'cloves', 'banana', 'oats', 'zest', 'pecans', 'juice', 'squares', 'flowers', 'applesauce', 'seeds', 'mayonnaise', 'margarine', 'sour cream', 'honey', 'potatoes', 'ginger', 'wheat germ', 'tofu', 'meal', 'allspice', 'chocolate chunks', 'dates', 'vinegar', 'coconut', 'starch', 'cardamom', 'beer', 'root', 'baking powder', 'walnuts', 'syrup', 'strawberries', 'blueberries', 'apple', 'papaya', 'quarters', 'vanilla', 'drained', 'diced', 'bran', 'gum', 'rum', 'toasted', 'flakes)', 'cherries', 'cubed', 'garnish', 'cereal', 'cranberries', 'separated', 'pieces', 'C)', 'nuts', 'cheese', 'tea', 'carrots', 'Delight®)', 'granules', 'seed', 'Splenda®)', 'Blend', 'zucchini', 'link)', 'thawed', 'molasses', 'puree', 'spice', 'Pam®)', 'Honey®)', 'drippings', 'bacon', 'cornmeal', 'lengthwise', 'whisked', 'Granulated)', 'Light)', 'Blend)', 'substitute', 'Arthur®)', 'Mixture)', 'peanuts', 'flavoring', 'Topping', 'pureed', 'Mill®)', 'Icing:', 'cognac', 'pumpkin', 'demerara', 'Oil', 'Extract', 'peeled', 'Applesauce', 'topping']

        # MILESTONE BROWNIES
        # self.foods = ['sugar', 'flour', 'eggs', 'chocolate', 'water', 'salt', 'butter', 'vanilla', 'oil', 'baking soda', 'baking powder']
        # self.foods = ["raspberr", "bourbon", "pecan", "strawberr", "hazelnut", "macadamia nut", "lime", "zucchini", "bacon", "soda", "apple", "coconut", "coffee", "cocoa", "sweet potato", "cake mix", "lentils", "lemon juice", "corn syrup", "maple syrup", "peppermint", "vanilla", "vanilla pudding mix", "tea", "shortening", "yogurt", "cherr", "water", "salad oil", "cooking oil", "canola oil", "olive oil", "flaxseed oil", "almond extract", "cornstarch", "cashew", "pumpkin pie spice", "coffee gran", "avocado", "raisin", "applesauce", "yellow cake mix", "honey", "rum", "milk", "almond milk", "condensed milk", "soy milk", "evaporated milk", "sour milk", "coconut milk", "ice cream", "protein powder", "heavy cream", "sour cream", "ice cream cone", "whipping cream", "liqueur", "cream of tartar", "caramel", "chocolate frosting", "nesquik", "nutella", "oreo", "sweet chocolate", "bittersweet chocolate", "german sweet chocolate", "toffee", "chocolate cake mix", "chocolate pudding mix", "white chocolate", "dark chocolate", "baking chocolate", "milk chocolate", "unsweetened chocolate", "cream cheese", "margarine", "vegan margarine", "chocolate chip", "bittersweet chocolate chip", "dark chocolate chip", "vegan chocolate chip", "chocolate malt powder", "thin mints", "white chocolate chip", "milk chocolate chip", "bitterswet chocolate", "butter", "butter flavored shortening", "butterscotch-flavored chips", "almond butter", "butterfinger", "buttermilk", "egg", "egg replacer", "egg yolk", "flour", "pastry flour", "spelt flour", "cooking spray", "self-rising flour", "sorghum flour", "teff flour", "tapioca flour", "almond flour", "wheat flour", "coconut flour", "rice flour", "barley flour", "potato flour", "cake flour", "sugar", "superfine sugar", "instant pudding", "german chocolate cake mix", "banana", "vodka", "egg substitute", "coconut oil", "vegetable oil", "marshmallow", "cinnamon", "food coloring", "chocolate syrup", "walnut", "salt", "peanut", "almond", "coarse salt", "black bean", "salted cashew", "sea salt", "salted butter", "unsalted butter", "stevia", "rice cereal", "cocoa powder", "artificial sweetener", "baking soda", "butterscotch chip", "carob powder", "egg white", "irish stout beer", "cake meal", "gluten-free all purpose baking flour", "butter or margarine", "baking powder", "peppermint extract", "skim milk", "brownie mix", "unsalted butter", "powedered peanut butter", "peanut butter", "confectioners' sugar", "granulated sugar", "brown sugar", "light brown sugar", "coconut sugar"]

        # SIMPLE BROWNIES
        self.foods = ["raspberr", "bourbon", "pecan", "strawberr", "hazelnut", "macadamia nut", "lime", "zucchini", "bacon", "soda", "apple", "coconut", "coffee", "cocoa", "sweet potato", "cake mix", "lentils", "lemon juice", "corn syrup", "maple syrup", "peppermint", "vanilla", "vanilla pudding mix", "tea", "shortening", "yogurt", "cherr", "water", "oil", "almond extract", "cornstarch", "cashew", "pumpkin pie spice", "avocado", "raisin", "applesauce", "yellow cake mix", "honey", "rum", "milk", "almond milk", "condensed milk", "soy milk", "evaporated milk", "sour milk", "coconut milk", "ice cream", "protein powder", "cream", "sour cream", "ice cream cone", "liqueur", "cream of tartar", "caramel", "nesquik", "nutella", "oreo", "toffee", "cream cheese", "margarine", "vegan margarine", "thin mint", "butter", "almond butter", "butterfinger", "buttermilk", "egg", "egg yolk", "egg white", "instant pudding", "banana", "vodka", "marshmallow", "cinnamon", "food color", "walnut", "salt", "peanut", "almond", "black bean", "stevia", "rice cereal", "artificial sweetener", "baking soda", "carob", "irish stout beer", "cake meal", "baking powder", "peppermint extract", "brownie mix", "peanut butter", "sugar", "confectioners' sugar", "brown sugar", "cooking spray", "white chocolate", "dark chocolate", "chocolate", "butterscotch chip", "chocolate malt powder", "chocolate syrup", "chocolate frosting", "chocolate cake mix", "chocolate pudding mix", "flour"]

        # FULL LASAGNA (note: sugar = white sugar)
        # self.foods = ["spaghetti sauce", "pasta sauce", "sun-dried tomato", "tomato", "crushed tomato", "diced tomato", "marinara sauce", "alfredo sauce", "flour", "parmesan", "mozzarella", "cheddar cheese", "feta", "ricotta cheese", "swiss cheese", "cottage cheese", "blue cheese", "monterey jack cheese", "romano cheese", "cream cheese", "cheese ravioli", "vegan cheese", "provolone cheese", "colby cheese", "goat cheese", "american cheese", "reggiano cheese", "mexican cheese", "evaporated milk", "milk", "eggplant", "egg", "tomato sauce", "tomato paste", "beef broth", "ground beef", "beef", "chuck", "steak", "zucchini", "garlic powder", "garlic", "garlic salt", "lasagna noodle", "lasagne noodle", "wide noodle", "lasagna sheet", "walnuts", "salt and pepper", "salt and ground black pepper", "salt and black pepper", "pepper", "black pepper", "ground black pepper", "bell pepper", "pepperoni", "jalapeno", "red pepper", "cayenne", "olives", "oil", "olive oil", "canola oil", "vegetable oil", "squash", "butternut squash", "butter", "salt", "margarine", "spinach", "onion", "pesto", "basil", "oregano", "nutmeg", "cream of mushroom soup", "mushroom", "parsley", "rosemary", "broccoli", "artichoke", "water", "sausage", "chicken broth", "vegetable broth", "cream of chicken soup", "chicken", "tofu", "carrot", "sour cream", "heavy cream", "whipped cream", "whipping cream", "thyme", "cumin", "bacon", "shrimp", "crab", "cinnamon", "sugar", "brown sugar", "sage", "celery", "pork", "turkey"]

        self.samefoods = [
            ["spaghetti sauce", "pasta sauce"],
            ["tomato", "crushed tomato", "diced tomato"],
            ["beef", "chuck", "steak", "ground beef"],
            ["lasagna noodle", "wide noodle", "lasagna sheet"],
            ["salt and pepper", "salt and ground black pepper", "salt and black pepper"],
            ["black pepper", "pepper", "ground black pepper"],
            ["red pepper", "cayenne"],
            ["olive oil", "oil"],
            ["squash", "butternut squash"],
            ["whipped cream", "whipping cream", "heavy cream"]
        ]

        self.foods = sorted(self.foods, key=len)
        self.foods.reverse();
        self.foods = self.normalizeStrings(self.foods)
        self.samefoods = [self.normalizeStrings(f) for f in self.samefoods]
        if debug:
            print(self.units)

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
            if self.debug:
                print("Unable to find unit in: " + ingredient)
            return quantity

    def equivalentFood(self, food):
        for group in self.samefoods:
            if food in group:
                # if self.debug:
                #     print("Equivalent foods: " + food + " = " + group[0])
                return group[0]
        return food

    # Determine what food is present in an ingredient
    def getFood(self, ingredient):
        for food in self.foods:
            if food in self.normalizeStrings([ingredient])[0]:
                return self.equivalentFood(food)
        if self.debug:
            print("Unable to find food in: " + ingredient)
        return ''

    # Parse single ingredient token
    def parseIngredient(self, ingredient):
        food = self.getFood(ingredient)
        amount = self.getUnitlessQuantity(ingredient, food, self.getQuantity(ingredient), self.getUnit(ingredient))
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
