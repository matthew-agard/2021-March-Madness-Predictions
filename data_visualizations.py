import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_yearly_base_rates(df):
    yearly_outcomes = df.groupby(['Year', 'Underdog_Upset']).agg({'Round': 'count'})
    yearly_games = df.groupby('Year').agg({'Round': 'count'})
    yearly_outcomes = pd.merge(yearly_outcomes, yearly_games, left_index=True, right_index=True)

    yearly_fave_wins = yearly_outcomes.loc[(slice(None), 0), :]
    yearly_base_rates = np.round(yearly_fave_wins['Round_x'] / yearly_fave_wins['Round_y'], 3)
    yearly_base_rates.index = yearly_base_rates.index.get_level_values(0)

    return yearly_base_rates


def get_seed_pairs(df):
    sorted_pairs = []

    for index, data in df.iterrows():
        sorted_pair = tuple(sorted([data['Seed_Favorite'], data['Seed_Underdog']]))
        sorted_pairs.append(sorted_pair)

    seed_pairs = pd.DataFrame(data = {
        'Round': df['Round'],
        'Pairs': sorted_pairs,
        'Underdog_Upset': df['Underdog_Upset']
    })

    return seed_pairs


def format_plot(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.tight_layout()