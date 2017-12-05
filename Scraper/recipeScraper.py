# With help form https://github.com/hhursev/recipe-scraper
from urllib import request
from bs4 import BeautifulSoup
from scraperUtils import normalize_string

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
}

class recipeScraper():

    def __init__(self, url):
        self.url = url
        self.soup = BeautifulSoup(request.urlopen(
                request.Request(url,
                headers=HEADERS)).read(), "html.parser")

    def url(self):
        return self.url

    def host(self):
        return 'allrecipes.com'

    def title(self):
        return self.soup.find('h1').get_text()

    def ingredients(self):
        ingredients_html = self.soup.findAll('li', {'class': "checkList__line"})

        return [
            normalize_string(ingredient.get_text().replace("ADVERTISEMENT", ""))
            for ingredient in ingredients_html
            if ingredient.get_text(strip=True) not in ('Add all ingredients to list', '', 'ADVERTISEMENT')
        ]

    def instructions(self):
        instructions_html = self.soup.findAll('span', {'class': 'recipe-directions__list--item'})

        return '\n'.join([
            normalize_string(instruction.get_text())
            for instruction in instructions_html
        ])

    def review_count(self):
        return int(self.soup.find('meta', {'itemprop': 'reviewCount'})['content'])

    def rating_stars(self):
        return float(self.soup.find('meta', {'itemprop': 'ratingValue'})['content'])

    def servings(self):
        return float(self.soup.find('meta', {'itemprop': 'recipeYield'})['content'])
