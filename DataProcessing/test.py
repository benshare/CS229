import json

# data is a dictionary with keys 'name', 'rating', 'reviews', 'servings', and 'ingredients'.
# Values are vectors (ingredients is a vector of vectors)
# Corresponding recipes have the same index in all values
def main():
    data = None
    _data = None
    with open('Processed Recipes/_Brownies.json') as data_file:
        _data = json.load(data_file)
    with open('Raw Recipes/Brownies.json') as data_file:
        data = json.load(data_file)

    keys = ['name', 'rating', 'reviews', 'servings', 'ingredients']
    _keys = ['name', 'rating', 'reviews', 'servings', '_ingredients']
    for i in range(len(keys)-1):
        assert len(data[keys[i]]) == len(data[keys[i+1]])

    for i in range(len(keys)-1):
        assert len(_data[_keys[i]]) == len(_data[_keys[i+1]])

    assert(len(data[keys[0]]) == len(_data[keys[0]]))
    l = len(data[keys[0]])
    print("Read",l,"recipes.")

    for i in range(l):
        print(data['ingredients'][i])
        print(_data['_ingredients'][i])

main()
