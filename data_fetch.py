import pandas as pd
from web_scraper_types import bs4_web_scrape, pandas_web_scrape

def get_team_data(url, attrs, header=1):
    teams_df = pandas_web_scrape(url, attrs, header)
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

    return coaches_df