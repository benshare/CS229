import json  # Reading data
from filterRecipes import RecipeFilterer
from parseIngredients import IngredientParser

# Parses all ingredients, then filters out recipes according to params in filterRecipes.
# To run: change variable filename, output data will be _filename
if __name__ == "__main__":
    # Read data
    data = None
    filename = "Brownies.json"
    tag = "_tiny_"
    inpath = "Raw Recipes/"
    outpath = "Processed Recipes/"
    cleanFoodFilename = "Ingredient Lists/Brownies_Ingredients_Clean_Tiny.json"

    with open(inpath + filename) as data_file:
        data = json.load(data_file)

    # Process data
    print("Parsing Ingredients...")
    data_ing = IngredientParser(data, cleanFoodFilename, debug=False).parseAll()
    print("Filtering Recipes...")
    data_out = RecipeFilterer(data_ing).filterAll()

    # Save data
    jsonFile = open(outpath + tag + filename, "w")
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
