import json

def main():
    with open('Lasagna.json') as data_file:
        data = json.load(data_file)

        clean_names_file = open('Lasagna_Ingredients_Clean_Full.json', 'r')
        clean_names = json.load(clean_names_file)

        # data is a dictionary with keys 'name', 'rating', 'reviews', and 'ingredients'.
        # Values are vectors (ingredients is a vector of vectors)
        # Corresponding recipes have the same index in all values
        
        for i in range(0, len(data['name'])):
            for j in range(0, len(data['ingredients'][i])):
                name = match_ingredient(data['ingredients'][i][j], clean_names)
                if name is not '':
                    print('Clean ingredient name: ',name) # Clean ingredient match here!

def excluded(raw_ingredient, exclude_list):
    for s in exclude_list:
        if s in raw_ingredient.lower():
            return True

    return False

def found_better_match(s_clean, raw_ingredient, best_match, exclude_list):
    return s_clean in raw_ingredient.lower() and len(s_clean) > len(best_match) \
       and not excluded(raw_ingredient, exclude_list)

def match_ingredient(raw_ingredient, parsed_names):
    best_match = ''
    for ing in parsed_names['ingredients']:
        if isinstance(ing, list): # Multiple names for same ingredient
            for s in ing:
                if found_better_match(s, raw_ingredient, best_match, parsed_names['exclude']):
                    best_match = ing[0]
        else:
            if found_better_match(ing, raw_ingredient, best_match, parsed_names['exclude']):
                best_match = ing
    
    return best_match

main()
