import json

# data is a dictionary with keys 'name', 'rating', 'reviews', 'servings', and 'ingredients'.
# Values are vectors (ingredients is a vector of vectors)
# Corresponding recipes have the same index in all values
def main():
    data = None
    _data = None
    with open('Raw Recipes/Brownies.json') as data_file:
        data = json.load(data_file)
    with open('Processed Recipes/_Brownies.json') as data_file:
        _data = json.load(data_file)

    keys = ['name', 'rating', 'reviews', 'servings', 'ingredients']
    _keys = ['name', 'rating', 'reviews', 'servings', '_ingredients']
    for i in range(len(keys)-1):
        assert len(data[keys[i]]) == len(data[keys[i+1]])

    for i in range(len(keys)-1):
        assert len(_data[_keys[i]]) == len(_data[_keys[i+1]])

    # assert(len(data[keys[0]]) == len(_data[keys[0]]))
    l = len(_data[keys[0]])
    print(len(data[keys[0]]),"raw recipes.")
    print(l,"filtered recipes.")

    # for i in range(l):
        # print(data['ingredients'][i])
        # print(_data['_ingredients'][i])

    # seenNames = {}
    # for i in range(l):
    #     if data['name'][i] in seenNames.keys():
    #         i2 = seenNames[data['name'][i]]
    #         if data['ingredients'][i] == data['ingredients'][i2]:
    #             print(i, seenNames[data['name'][i]], data['name'][i])
    #             print(data['ingredients'][i])
    #             print(data['ingredients'][i2])
    #             print(data['servings'][i])
    #             print(data['servings'][i2])
    #             print(data['rating'][i])
    #             print(data['rating'][i2])
    #             print(data['reviews'][i])
    #             print(data['reviews'][i2])
    #     else:
    #         seenNames[data['name'][i]] = i

main()
