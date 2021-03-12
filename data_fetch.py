import pandas as pd
import re
from web_scraper_types import bs4_web_scrape, pandas_web_scrape, bracket_web_scrape

def get_team_data(url, attrs, header=1):
    try:
        teams_df = pandas_web_scrape(url, attrs, header)
    except ValueError:
        teams_df = [pd.DataFrame()]
    
    return teams_df[0]


def get_rankings_data(url):
    raw_html = bs4_web_scrape(url, attrs={"id": "ratings"})
    rankings_df = pd.DataFrame(columns=['Team', 'Top_25'])

    for i in range(len(raw_html)):
        if raw_html[i].find('a'):
            team = raw_html[i].find("a")
            rankings_df.loc[i] = [team.text, 1 if (len(rankings_df) < 25) else 0]
            
    return rankings_df


def get_coach_data(url):
    raw_html = bs4_web_scrape(url, attrs={"id": "coaches"})
    coaches_df = pd.DataFrame(columns=['Coach_Team', 'MM', 'S16', 'F4', 'Champs'])

    for i in range(len(raw_html)):
        if(raw_html[i].find('a')):
            coach_team = raw_html[i].find_all("a")[1]
            mm_apps = raw_html[i].find("td", attrs={"data-stat": "ncaa_car"})
            sw16_apps = raw_html[i].find("td", attrs={"data-stat": "sw16_car"})
            f4_apps = raw_html[i].find("td", attrs={"data-stat": "ff_car"})
            champ_wins = raw_html[i].find("td", attrs={"data-stat": "champ_car"})

            coaches_df.loc[i] = [coach_team.text, mm_apps.text, sw16_apps.text, f4_apps.text, champ_wins.text]

    return coaches_df.drop_duplicates(subset='Coach_Team', keep='last')


def get_current_bracket(url):
    raw_html = bracket_web_scrape(url, attrs={"id": "bracket"})
    current_bracket = pd.DataFrame(columns=['Seed', 'Team', 'Seed.1', 'Team.1'])

    for i, game in enumerate(raw_html):
        game_string = game.find('dt')
        
        teams = [name.text for name in game_string.find_all('a')]
        
        seeds = re.findall(r'\d+', game_string.text) 
        seeds = list(map(int, seeds))
        
        current_bracket.loc[i] = [seeds[0], teams[0], seeds[1], teams[1]]

    return current_bracket