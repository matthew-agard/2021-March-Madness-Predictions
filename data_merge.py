"""Data Merge Helper Functions

This script is used as a helper module in the data_pipeline script.
The following functions are present:
    * merge_clean_team_stats
    * merge_clean_rankings
    * merge_clean_coaches
    * merge_clean_tourney_games

Requires a minimum of the 'pandas' library being installed in your environment to run.
"""

import pandas as pd


def merge_clean_team_stats(basic_df, adv_df):
    """Merge basic and advanced season stats (by team)

    Parameters
    ----------
    basic_df : DataFrame
        Teams' basic regular season stats
    adv_df : DataFrame
        Teams' advanced regular season stats

    Returns
    -------
    season_team_stats_df : DataFrame
        All teams' regular season stats
    """
    # Merge on the school name
    season_team_stats_df = pd.merge(basic_df, adv_df, on='School')

    # Strip the 'NCAA' tag from the teams with a tournament berth
    if season_team_stats_df['School'].str.contains('NCAA').any():
        season_team_stats_df['School'] = season_team_stats_df['School'].apply(lambda school: school[:-5])
    
    return season_team_stats_df


def merge_clean_rankings(team_stats_df, rankings_df):
    """Merge season rankings with season stats

    Parameters
    ----------
    team_stats_df : DataFrame
        All teams' regular season stats
    rankings_df : DataFrame
        Teams' regular season rankings stats

    Returns
    -------
    season_stats_rankings_df : DataFrame
        Newly-merged DataFrame of a teams' regular season stats with their regular season ranking
    """
    # Merge on the school name
    season_stats_rankings_df = pd.merge(team_stats_df, rankings_df, 
                                        left_on='School', right_on='Team').drop('Team', axis=1)
    
    return season_stats_rankings_df


def merge_clean_coaches(stats_rankings_df, coaches_df):
    """Merge coach performance to teams' season stats and rankings

    Parameters
    ----------
    stats_rankings_df : DataFrame
        All teams' regular season stats and rankings
    coaches_df : DataFrame
        Teams' coach performances historically in the tournament

    Returns
    -------
    all_season_stats_df : DataFrame
        Newly-merged DataFrame of a teams' regular season stats, 
        regular season ranking, and coach performance
    """
    # Merge on the school name
    all_season_stats_df = pd.merge(stats_rankings_df, coaches_df,
                                    left_on='School', right_on='Coach_Team').drop('Coach_Team', axis=1)

    return all_season_stats_df


def merge_clean_tourney_games(mm_df, all_season_df):
    """Merge all team data onto tournament matchups DataFrame

    Parameters
    ----------
    mm_df : DataFrame
        All teams' tournament matchups
    all_season_df : DataFrame
        All teams' data

    Returns
    -------
    all_data_df : DataFrame
        Completed dataset
    """
    # Merge favorites' season data onto tournament matchups DataFrame
    favorites_data_df = pd.merge(mm_df, all_season_df, 
                                left_on='Team_Favorite', right_on='School').drop('School', axis=1)
    
    # Merge underdogs' season data onto tournament matchups DataFrame
    # Account for duplicate stats column names with suffix labeling
    all_data_df = pd.merge(favorites_data_df, all_season_df, suffixes=("_Favorite", "_Underdog"),
                            left_on='Team_Underdog', right_on='School').drop('School', axis=1)

    return all_data_df