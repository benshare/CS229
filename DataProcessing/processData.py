import json  # Reading data
from filterRecipes import RecipeFilterer
from parseIngredients import IngredientParser

# Parses all ingredients, then filters out recipes according to params in filterRecipes.
# To run: change variable filename, output data will be _filename
if __name__ == "__main__":
    # Read data
    data = None
    filename = 'Brownies.json'
    tag = "_Simple_"
    with open(filename) as data_file:
        data = json.load(data_file)

    # Process data
    print("Parsing Ingredients...")
    data_ing = IngredientParser(data).parseAll()
    print("Filtering Recipes...")
    data_out = RecipeFilterer(data_ing).filterAll()

    # Save data
    jsonFile = open(tag + filename, "w")
    jsonFile.truncate()
    jsonFile.write(json.dumps({
            "name": data_out['name'],
            "rating": data_out['rating'],
            "reviews": data_out['reviews'],
            "_ingredients": data_out['_ingredients']}))
    jsonFile.write("\n")
    jsonFile.close()

    print('Done!')
