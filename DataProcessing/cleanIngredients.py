import json

def main():
    with open('Brownies.json') as data_file:
        data = json.load(data_file)

        clean_names_file = open('Brownies_Ingredients_Clean.json', 'r')
        clean_names = json.load(clean_names_file)

        # data is a dictionary with keys 'name', 'rating', 'reviews', and 'ingredients'.
        # Values are vectors (ingredients is a vector of vectors)
        # Corresponding recipes have the same index in all values
        
        for i in range(0, len(data['name'])):
            for j in range(0, len(data['ingredients'][i])):
                print('Ingredient name: ', match_ingredient(data['ingredients'][i][j], clean_names['ingredients']))

def match_ingredient(raw_ingredient, parsed_names):
    best_match = ''
    for s in parsed_names:
        if s in raw_ingredient.lower() and len(s) > len(best_match):
            best_match = s
    
    return best_match

main()
