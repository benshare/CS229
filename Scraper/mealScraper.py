# With help form https://github.com/hhursev/recipe-scraper
from urllib import request  # make requests
import urllib.error  # error handling
from bs4 import BeautifulSoup  # parse HTML

import re  # regular expressions
import json  # save data
import time  # time.sleep to avoid too many requests

from recipeScraper import recipeScraper  # scrape individual recipes
from scraperUtils import normalize_string

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
}

class AllrecipesMealScraper():

    def __init__(self, url):
        self.url = url  # URL to a the default category page (should end with something like ...cardslot%201)
        self.delay = 1.5  # Seconds between requests

    def url(self):
        return self.url

    def host(self):
        return 'allrecipes.com'

    def title(self):
        return self.soup.find('h1').get_text()

    # Scrapes data about recipes for all pages of a category.
    def scrape(self, filename):
        self.filename = filename + '.json'
        self.title = []
        self.rating = []
        self.numReviews = []
        self.ingredients = []
        self.servings = []
        self.recipeURLs = set()
        self.count = 0

        # Iterates through pages of recipes
        page = 1
        while(True):
            self.soup = None

            try:
                # Are there more pages? (infinite scroll)
                url = self.url + '&page=' + str(page) + '#' + str(page)
                print('Processing:', url)
                self.soup = BeautifulSoup(request.urlopen(
                        request.Request(url,
                        headers=HEADERS)).read(), "html.parser")
                page += 1
                time.sleep(self.delay);

            except urllib.error.HTTPError as e:
                assert (e.code == 404)  # Out of pages!
                print('Found', page-1, 'pages.')

                # Scrape all recipes that have been found
                self.count = 0
                for recipeURL in self.recipeURLs:
                    self.tryScrapingRecipe(recipeURL, 0)
                print('Found', self.count, 'recipes')

                self.saveProgress()
                print('Done!')
                return

            # Gather recipes found so far
            recipes_html = self.soup.findAll('a', href=True)
            self.recipeURLs = self.recipeURLs.union(set([
                'http://' + self.host() + link['href']
                for link in recipes_html if link['href'].startswith('/recipe/')]))

    # Write recipes so far to a json file
    def saveProgress(self):
        self.jsonFile = open(self.filename, "w")
        self.jsonFile.truncate()
        self.jsonFile.write(json.dumps({
                "name": self.title,
                "rating": self.rating,
                "reviews": self.numReviews,
                "servings": self.servings,
                "ingredients": self.ingredients}))
        self.jsonFile.write("\n")
        self.jsonFile.close()

    # Tries to crawl a recipe.
    # Recursively calls self if error occurs a maximum of 10 tmes before moving on.
    # Saves progress after every recipe.
    def tryScrapingRecipe(self, recipeURL, tries):
        if (tries > 10):
            print("***Skipping recipe")
            return

        print(self.count+1, "- Scraping:", recipeURL)
        try:
            scrape = recipeScraper(recipeURL)
            self.title.append(scrape.title())
            self.rating.append(scrape.rating_stars())
            self.numReviews.append(scrape.review_count())
            self.servings.append(scrape.servings())
            self.ingredients.append(scrape.ingredients())

            self.saveProgress();
            self.count += 1
            time.sleep(self.delay);
        except urllib.error.HTTPError as e:
            print('***Caught HTTPError')
            time.sleep(self.delay*5);
            self.tryScrapingRecipe(recipeURL, tries+1)
        except urllib.error.URLError as e:
            print('***Caught URLError')
            time.sleep(self.delay*5);
            self.tryScrapingRecipe(recipeURL, tries+1)

        # except UnicodeEncodeError:
        #     print('UnicodeEncodeError')
        #     return
        # except ConnectionResetError:
        #     print('ConnectionResetError')
        #     time.sleep(5);
        #     self.tryScrapingRecipe(recipeURL)


def main():
    # AllrecipesMealScraper('http://allrecipes.com/recipes/343/bread/quick-bread/fruit-bread/banana-bread/?internalSource=hubcard&referringContentType=search%20results&clickId=cardslot%201').scrape('BananaBread')
    # AllrecipesMealScraper('http://allrecipes.com/recipes/839/desserts/cookies/chocolate-chip-cookies/?internalSource=hub%20nav&referringId=17254&referringContentType=recipe%20hub&referringPosition=5&linkName=hub%20nav%20exposed&clickId=hub%20nav%205').scrape('ChocChipCookies')
    AllrecipesMealScraper('http://allrecipes.com/recipes/502/main-dish/pasta/lasagna/?internalSource=hubcard&referringContentType=search%20results&clickId=cardslot%201').scrape('Lasagna')
    # AllrecipesMealScraper('http://allrecipes.com/recipes/151/breakfast-and-brunch/pancakes/?internalSource=hubcard&referringContentType=search%20results&clickId=cardslot%201').scrape('Pancakes')
    # AllrecipesMealScraper('http://allrecipes.com/recipes/838/desserts/cookies/brownies/?internalSource=hubcard&referringContentType=search%20results&clickId=cardslot%201').scrape('Brownies')



if __name__ == "__main__":
    main()
