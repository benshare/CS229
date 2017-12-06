import json

# data is a dictionary with keys 'name', 'rating', 'reviews', 'servings', and 'ingredients'.
# Values are vectors (ingredients is a vector of vectors)
# Corresponding recipes have the same index in all values
def main():
    with open('Processed Recipes/_Brownie.json') as data_file:
        data = json.load(data_file)
        keys = ['name', 'rating', 'reviews', 'servings', '_ingredients']
        for i in range(len(keys)-1):
            assert len(data[keys[i]]) == len(data[keys[i+1]])

        l = len(data[keys[0]])
        print("Read",l,"recipes.")

        for i in range(l):
            print(data['name'][i],"-",data['_ingredients'][i])

main()
