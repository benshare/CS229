import json

def main():
    with open('BananaBread.json') as data_file:
        data = json.load(data_file)

        # data is a dictionary with keys 'name', 'rating', 'reviews', and 'ingredients'.
        # Values are vectors (ingredients is a vector of vectors)
        # Corresponding recipes have the same index in all values
        recipe = 0
        for key in ['name', 'rating', 'reviews', 'ingredients']:
            print(key,'-',data[key][0])

main()
