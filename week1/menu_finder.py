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
    list_items_dishes = soup.findAll(True, {'class':[re.compile(r'item-header'), re.compile(r"menu-item .*")]})
    # Store menu items into list comprehension and filter items with removing reviews
    list_items_with_days = [item for item in list_items_dishes if not "item-review" in item.find_parent("div")["class"]]
    
    # Format printing with adding new line after each day and remove extra spaces
    for item in list_items_with_days:
        if "item-header" in item.get("class"):
            print()
            print(item.get_text())
        else:
            string_item = re.sub(' +', ' ', item.get_text(separator=' '))
            print(string_item)
    
main()
