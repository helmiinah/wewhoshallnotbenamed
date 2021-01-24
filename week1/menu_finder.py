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
    list_items_dishes = soup.findAll(True, {'class': [re.compile(r'item-header'), re.compile(r"menu-item item.*")]})

    # Create a regex pattern to match a date format found in dish headers:
    date_pattern = re.compile(r".*\d+\.\d+\..*")

    # Store menu items into list comprehension and filter out review headers using the date pattern:
    list_items_with_days = [item for item in list_items_dishes if "item-header" not in item.get("class")
                            or re.match(date_pattern, item.get_text())]

    # Filter out irrelevant captions from menu item bodies
    # (2nd child node of a menu item includes dish name & tags):
    # (this could probably be done as a part of the previous list comprehension)
    list_items_with_days = [item.findChildren()[1] if "menu-item" in item.get("class")
                            else item for item in list_items_with_days]
    
    # Format printing with adding new line after each day and remove extra spaces
    for item in list_items_with_days:
        if "item-header" in item.get("class"):
            print()
            print(item.get_text())
        else:
            string_item = re.sub(' +', ' ', item.get_text(separator=' '))
            # Remove the strange square brackets from tags:
            string_item = re.sub('[\[\]]', '', string_item)
            print(string_item)
    
main()
