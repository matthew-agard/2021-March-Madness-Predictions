import pandas as pd
import requests
from bs4 import BeautifulSoup


def bs4_web_scrape(url, attrs):
    url = url
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    table = soup.find("table", attrs=attrs)

    rows = table.find_all("tr")
    return rows


def pandas_web_scrape(url, attrs, header):
    url = url
    arr = pd.read_html(url, attrs=attrs, header=header)

    return arr


def bracket_web_scrape(url, attrs):
    url = url
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    bracket = soup.find("div", attrs=attrs)

    games = bracket.find_all("dl")
    return games