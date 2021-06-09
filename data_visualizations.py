"""EDA Helper Functions

This script is used as a module in the March_Madness_Predictions Jupyter notebooks.
The following functions are present:
    * get_yearly_base_rates
    * get_seed_pairs
    * format_plot

Requires a minimum of the 'pandas', 'numpy', and 'matplotlib' libraries being present 
in your environment to run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_yearly_base_rates(df):
    """Calculates base rates (per year) to observe tournament outcome trends

    Parameters
    ----------
    df : DataFrame
        Set of all March Madness matchups
    
    Returns
    -------
    yearly_base_rates : DataFrame
        DataFrame of favorites' wins base rate (per year)
    """

    # Extract number of upsets, favorites' wins across all game outcomes (per year)
    yearly_outcomes = df.groupby(['Year', 'Underdog_Upset']).agg({'Round': 'count'})
    yearly_games = df.groupby('Year').agg({'Round': 'count'})
    yearly_outcomes = pd.merge(yearly_outcomes, yearly_games, left_index=True, right_index=True)

    # Calculate favorites' wins base rate; store & return in DataFrame
    yearly_fave_wins = yearly_outcomes.loc[(slice(None), 0), :]
    yearly_base_rates = np.round(yearly_fave_wins['Round_x'] / yearly_fave_wins['Round_y'], 3)
    yearly_base_rates.index = yearly_base_rates.index.get_level_values(0)

    return yearly_base_rates


def get_seed_pairs(df):
    """Format team seed pairs & game outcomes across all rounds

    Parameters
    ----------
    df : DataFrame
        Set of all March Madness matchups
    
    Returns
    -------
    seed_pairs : DataFrame
        DataFrame of team seeds and their game outcomes
    """
    sorted_pairs = []

    # Collect seed pairs of all tournament matchups
    for index, data in df.iterrows():
        # Store pairs in sorted order to achieve continuity (i.e. 6-11 seed matchup same as 11-6 seed)
        sorted_pair = tuple(sorted([data['Seed_Favorite'], data['Seed_Underdog']]))
        sorted_pairs.append(sorted_pair)

    # Return stored results in dataframe
    seed_pairs = pd.DataFrame(data = {
        'Round': df['Round'],
        'Pairs': sorted_pairs,
        'Underdog_Upset': df['Underdog_Upset']
    })

    return seed_pairs


def format_plot(title, xlabel, ylabel):
    """Format EDA plots with this reusable function

    Parameters
    ----------
    title : str
        Plot title
    xlabel : str
        X-axis title
    ylabel : str
        Y-axis title
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.tight_layout()