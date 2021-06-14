"""Data Fetch Helper Functions

This script is used as a helper module in the data_pipeline script; 
also used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * get_team_data
    * get_rankings_data
    * get_coach_data
    * get_null_rows
    * get_feature_null_counts
    * get_current_bracket

Requires a minimum of the 'pandas' and 're' libraries, as well as the 
'web_scraper_types' helper module, being present in your environment to run.
"""

import pandas as pd
import re
from web_scraper_types import bs4_web_scrape, pandas_web_scrape, bracket_web_scrape


def get_team_data(url, attrs, header=1):
    """Fetch team data (season stats, historical tournament performance)

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        Characteristics to idenitfy HTML element of interest
    header : int, optional
        Row in raw data to use for column headers (default=1)

    Returns
    -------
    teams_df[0] : DataFrame
        Web-scraped data points read into a DataFrame
    """
    try:
        # Read team data into dataframe
        teams_df = pandas_web_scrape(url, attrs, header)
    except ValueError:
        # Catch error with empty DataFrame is requested team data doesn't exist
        teams_df = [pd.DataFrame()]
    
    return teams_df[0]


def get_rankings_data(url):
    """Fetch team season rankings

    Parameters
    ----------
    url : str
        URL path to data

    Returns
    -------
    rankings_df : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw data and prepare DataFrame
    raw_html = bs4_web_scrape(url, attrs={"id": "ratings"})
    rankings_df = pd.DataFrame(columns=['Team', 'Top_25'])

    # Iterate over raw data to extract team and rank HTML elements
    for i, rank in enumerate(raw_html):
        if rank.find('a'):
            team = rank.find('a')
            """CONSIDER 'AP Rank' FEATURE ON SPORTS REFERENCE"""
            # Identify Top 25 teams using ternary operator to produce binary output
            rankings_df.loc[i] = [team.text, 1 if (len(rankings_df) < 25) else 0]
            
    return rankings_df


def get_coach_data(url):
    """Fetch team coach performance

    Parameters
    ----------
    url : str
        URL path to data

    Returns
    -------
    coaches_df : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw data and prepare DataFrame
    raw_html = bs4_web_scrape(url, attrs={"id": "coaches"})
    coaches_df = pd.DataFrame(columns=['Coach_Team', 'MM', 'S16', 'F4', 'Champs'])

    # Iterate over raw data to extract coach tournament appearances HTML elements
    for i, row in enumerate(raw_html):
        if(row.find('a')):
            coach_team = row.find_all('a')[1]
            mm_apps = row.find("td", attrs={"data-stat": "ncaa_car"})
            sw16_apps = row.find("td", attrs={"data-stat": "sw16_car"})
            f4_apps = row.find("td", attrs={"data-stat": "ff_car"})
            champ_wins = row.find("td", attrs={"data-stat": "champ_car"})

            coaches_df.loc[i] = [coach_team.text, mm_apps.text, sw16_apps.text, f4_apps.text, champ_wins.text]

    return coaches_df.drop_duplicates(subset='Coach_Team', keep='last')


def get_null_rows(null_fills, df):
    """Fetch rows with any nulls; used for imputing new values

    Parameters
    ----------
    null_fills : list
        Collection of features where nulls reside
    df : DataFrame
        Fully merged dataset

    Returns
    -------
    DataFrame
        Cross-section of df; contains the rows where nulls reside for features in null_fills list
    """
    rows = df[df[null_fills].isnull().any(axis=1)]
    return rows[['Year'] + null_fills]


def get_feature_null_counts(df):
    """Count number of nulls for each feature containing any nulls

    Parameters
    ----------
    df : DataFrame
        Fully merged dataset

    Returns
    -------
    DataFrame
        Structure containing all features with nulls, sorted in descending order by their number of nulls
    """
    nulls = df.isnull().sum().sort_values(ascending=False)
    return nulls[nulls > 0]


def get_current_bracket(url):
    """Fetch current tournament bracket matchups

    Parameters
    ----------
    url : str
        URL path to data

    Returns
    -------
    current_bracket : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw data and prepare DataFrame
    raw_html = bracket_web_scrape(url, attrs={"id": "bracket"})
    current_bracket = pd.DataFrame(columns=['Seed', 'Team', 'Seed.1', 'Team.1'])

    # Iterate over raw data to extract team and their seeds
    for i, game in enumerate(raw_html):
        game_string = game.find('dt')

        teams = [name['title'] for name in game_string.find_all('a')]

        seeds = re.findall(r'\d+', game_string.text) 
        seeds = list(map(int, seeds))

        try:
            # Read team matchups into dataframe
            current_bracket.loc[i] = [seeds[0], teams[0], seeds[1], teams[1]]
        except IndexError:
            # Catch error where 1st Round awaits First Four winners
            if len(teams) > 0:
                current_bracket.loc[i] = [seeds[0], teams[0], 0, None]
                
    current_bracket.index = range(len(current_bracket))
    return current_bracket