"""Web Scraper Helper Functions

This script is used as a helper module in the data_fetch script.
The following functions are present:
    * pandas_web_scrape
    * bs4_web_scrape
    * bracket_web_scrape

Requires a minimum of the 'pandas', 'requests', and 'BeautifulSoup' libraries being present 
in your environment to run.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup


def pandas_web_scrape(url, attrs, header):
    """Pandas web scraper

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        characteristics to idenitfy HTML element of interest
    header : int
        row in raw data to use for column headers

    Returns
    -------
    arr : list
        Collection of all webpage data points (by row)
    """
    # Configure scraper and get table data
    arr = pd.read_html(url, attrs=attrs, header=header)
    return arr


def bs4_web_scrape(url, attrs):
    """BeautifulSoup table web scraper

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        characteristics to idenitfy HTML element of interest

    Returns
    -------
    rows : list
        Collection of all webpage data points (by row)
    """
    # Configure scraper
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    # Find table and get its data
    table = soup.find("table", attrs=attrs)
    rows = table.find_all("tr")

    return rows


def bracket_web_scrape(url, attrs):
    """BeautifulSoup bracket web scraper

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        characteristics to idenitfy HTML element of interest

    Returns
    -------
    games : list
        Collection of all tournament game data points (by row)
    """
    # Configure scraper
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    # Find bracket and get its data
    bracket = soup.find("div", attrs=attrs)
    games = bracket.find_all("dl")

    return games