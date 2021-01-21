from urllib import request
from bs4 import BeautifulSoup
from pprint import pprint  # For formatted printing/ pretty printing
import re

######################################
# Script to extract the day's menu   #
# for Unicafe Kaivopiha              #
# from lounaat.info                  #
######################################


def main():
    # Download page as html
    url = 'https://www.lounaat.info/lounas/unicafe-ylioppilasaukio/helsinki'
    html = request.urlopen(url).read().decode('utf8')
    # transform into beautifulsoup element:
    soup = BeautifulSoup(html, 'html.parser')

    # Store all menu items (for every day available) to a list:
    list_items_dishes = soup.findAll("li", {"class": re.compile(r"menu-item .*")})
    for item in list_items_dishes:
        # pprint(item)  # get raw html
        pprint(item.get_text())  # get text only

    # next: extracting the days for which each menu item belongs to (or if we only want the menu for the current day?)
    # and formatting


main()
