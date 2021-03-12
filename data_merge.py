import pandas as pd
import numpy as np

def merge_clean_team_stats(basic_df, adv_df):
    season_team_stats_df = pd.merge(basic_df, adv_df, on='School')
    season_team_stats_df['School'] = season_team_stats_df['School'].apply(lambda school: school[:-5])
    
    return season_team_stats_df


def merge_clean_rankings(team_stats_df, rankings_df):
    season_stats_rankings_df = pd.merge(team_stats_df, rankings_df, 
                                        left_on='School', right_on='Team').drop('Team', axis=1)
    
    return season_stats_rankings_df


def merge_clean_coaches(stats_rankings_df, coaches_df):
    all_season_stats_df = pd.merge(stats_rankings_df, coaches_df,
                                    left_on='School', right_on='Coach_Team').drop('Coach_Team', axis=1)

    return all_season_stats_df


def merge_clean_tourney_games(mm_df, all_season_df):
    favorites_data_df = pd.merge(mm_df, all_season_df, 
                                left_on='Team_Favorite', right_on='School').drop('School', axis=1)

    all_data_df = pd.merge(favorites_data_df, all_season_df, suffixes=("_Favorite", "_Underdog"),
                            left_on='Team_Underdog', right_on='School').drop('School', axis=1)

    return all_data_df