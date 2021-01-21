from urllib import request
from pprint import pprint  # For formatted printing/ pretty printing
######################################
# Script to extract the day's menu   #
# for Unicafe Kaivopiha              #
# from lounaat.info                  #
######################################


def main():
    # Download page as html
    url = 'https://www.lounaat.info/lounas/unicafe-ylioppilasaukio/helsinki'
    html = request.urlopen(url).read().decode('utf8')
    # pprint(html)

    # transform into beautifulsoup element


main()
