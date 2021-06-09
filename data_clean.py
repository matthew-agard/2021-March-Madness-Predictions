"""Data Clean Helper Functions

This script is used as a helper module in the data_pipeline script.

The following functions are present:
    * clean_basic_stats
    * clean_adv_stats
    * clean_coach_stats
    * clean_merged_season_stats
    * clean_tourney_data
    * clean_round_cols
    * clean_curr_round_data
    * fill_playin_teams
    * clean_bracket

Requires a minimum of the 'pandas' and 'datetime' libraries, as well as the 'data_integrity' 
and 'feature_engineering' helper modules, being present in your environment to run.
"""

import pandas as pd
from datetime import datetime
from data_integrity import coach_to_season_dict, hist_season_to_tourney_dict, curr_season_to_tourney_dict
from feature_engineering import totals_to_game_average, create_faves_underdogs, bidirectional_rounds_str_numeric, create_target_variable

current_year = datetime.now().year


def clean_basic_stats(df):
    """Clean standard regular season stats

    Parameters
    ----------
    df : DataFrame
        Freshly scraped basic regular season data

    Returns
    -------
    ncaa_df : DataFrame
        All basic regular reason data for March Madness teams
    """
    # Remove useless "feature" used in table formatting
    useless_feats = ['Rk', 'MP'] + [col for col in df.columns 
                                    if ('Unnamed' in col) or ('W.' in col) or ('L.' in col)]
    # Remove linearly dependent features
    lin_dep_feats = ['W', 'L', 'SRS', 'FGA', '3PA', 'FTA']    
    df.drop(useless_feats + lin_dep_feats, axis=1, inplace=True)

    # Remove useless rows used in table formatting
    df = df[(df['School'] != 'School') & (df['G'] != 'Overall')]

    # Filter out teams that didn't participate in March Madness
    ncaa_df = df[df['School'].str.contains('NCAA')]

    return ncaa_df


def clean_adv_stats(df):
    """Clean advanced regular season stats

    Parameters
    ----------
    df : DataFrame
        Freshly scraped advanced regular season data

    Returns
    -------
    DataFrame
        All advanced regular reason data for March Madness teams
    """
    # Filter out redundant features already captured from basic stats web scraping
    return pd.concat([df['School'], df.iloc[:, -13:]], axis=1)


def clean_coach_stats(coach_df):
    """Clean coach tournament performance stats

    Parameters
    ----------
    coach_df : DataFrame
        Freshly scraped coach tournament data

    Returns
    -------
    coach_df : DataFrame
        Cleaned coach data for March Madness teams
    """
    # Change team names accordingly to ensure successful merging with team stats
    coach_df['Coach_Team'].replace(coach_to_season_dict, inplace=True)

    # Fill null values with '0' placeholder
    coach_df.iloc[:, 1:] = coach_df.iloc[:, 1:].replace('', '0')

    return coach_df


def clean_merged_season_stats(year, all_season_df, season_basic_df):
    """Clean fully merged dataset

    Parameters
    ----------
    year : int
        Calendar year
    all_season_df : DataFrame
        Complete regular season dataset (uncleaned)
    season_basic_df : DataFrame
        Cleaned basic regular season team stats (used for totals_to_game_average() function)
    
    Returns
    -------
    all_season_df : DataFrame
        Cleaned regular season dataset, ready for merging with tournament matchup data
    """
    # Change team names accordingly to ensure successful merging with tournament matchups
    all_season_df['School'].replace(
        hist_season_to_tourney_dict if (year != current_year) else curr_season_to_tourney_dict, 
        inplace=True
    )

    # Convert team regular season stats from season totals to per game averages
    totals_to_game_average(all_season_df, season_basic_df)

    return all_season_df


def clean_tourney_data(year, mm_df, season_df):
    """Clean tournament matchups data

    Parameters
    ----------
    year : int
        Calendar year
    mm_df : DataFrame
        Freshly scraped tournament matchup data
    season_df : DataFrame
        Cleaned basic regular season team stats (used for create_faves_underdogs() function)
    
    Returns
    -------
    mm_df : DataFrame
        Cleaned tournament matchup dataset, ready for merging with regular season stats data
    """
    # Properly format all names to ensure successful merging with regular season stats
    if year != current_year:
        for col in ['Round', 'Team', 'Team.1']:
            mm_df[col] = mm_df[col].apply(lambda name: name[:-(len(name) // 2)].strip())

    # Transform team listings into favorite-underdog matchups (using seeds & regular season record)
    faves_unds = create_faves_underdogs(mm_df, season_df)

    # Create new features representing favorite-underdog matchups
    mm_df_struct = ['Seed', 'Team', 'Score']
    for key in faves_unds.keys():    
        for j in range(len(mm_df_struct)):
            try:
                mm_df[mm_df_struct[j] + "_" + key] = faves_unds[key][:, j]
            except IndexError:
                continue
    # Create target variable (for training dataset only, otherwise KeyError is thrown)
    try:
        mm_df['Underdog_Upset'] = create_target_variable(mm_df)
    except KeyError:
        pass
            
    # Drop old matchup data features
    mm_df_drop = ['Seed', 'Team', 'Seed.1', 'Team.1']
    mm_df.drop(mm_df_drop, axis=1, inplace=True)

    return mm_df


def clean_round_cols(df):
    """Clean column names used for generating bracket rounds

    Parameters
    ----------
    df : DataFrame
        Dataset used for generating bracket round
    """
    df.rename(columns = {
        'Seed_Favorite': 'Seed',
        'Team_Favorite': 'Team',
        'Seed_Underdog': 'Seed.1',
        'Team_Underdog': 'Team.1',
    }, inplace=True)


def clean_curr_round_data(all_round_data, curr_X, school_matchups_df):
    """Format completed bracket predictions (for EDA)

    Parameters
    ----------
    all_round_data : DataFrame
        All tournament data
    curr_X : DataFrame
        Data used for model to predict matchup outcomes
    school_matchups_df : DataFrame
        Data used to store model prediction output
    
    Returns
    -------
    curr_X : DataFrame
        Data used for model to predict matchup outcomes (cleaned)
    school_matchups_df : DataFrame
        Data used to store model prediction output (cleaned)
    """
    # Assign proper seeds to round matchups; concatenate to the rest of the round's data
    curr_X[['Seed_Favorite', 'Seed_Underdog']] = all_round_data[['Seed_Favorite', 'Seed_Underdog']]
    curr_X = pd.concat([school_matchups_df, curr_X], axis=1)
    
    # Rename prediction data columns in preparation for next round
    clean_round_cols(curr_X)

    # Drop duplicate matchups; reindex to account for change in DataFrame size
    curr_X.drop_duplicates(subset=['Team', 'Team.1'], inplace=True)
    curr_X.index = range(len(curr_X))
    
    # Drop duplicate matchups; reindex to account for change in DataFrame size
    school_matchups_df.drop_duplicates(subset=['Team_Favorite', 'Team_Underdog'], inplace=True)
    school_matchups_df.index = range(len(school_matchups_df))

    return curr_X, school_matchups_df


def fill_playin_teams(all_curr_matchups):
    """Fill bracket nulls with play-in winners

    Parameters
    ----------
    all_curr_matchups : list
        DataFrames of matchups from each round
    """
    # Get first round matchups data
    first_round = all_curr_matchups[1]
    # Get indexes from rows where the play-in nulls are present
    playin_nulls = list(first_round[first_round.isnull().any(axis=1)].index)

    for i, data in all_curr_matchups[0].iterrows():
        # Get winner seed & name from play-in DataFrame
        winner_seed = data['Seed'] if (data['Underdog_Upset'] == 0) else data['Seed.1']
        winner_team = data['Team'] if (data['Underdog_Upset'] == 0) else data['Team.1']

        # Place play-in winner in appropriate first round matchup (based on playin_nulls index)
        first_round.loc[playin_nulls[i], 'Seed.1'] = winner_seed
        first_round.loc[playin_nulls[i], 'Team.1'] = winner_team


def clean_bracket(all_curr_matchups, all_curr_rounds):
    """Format completed bracket predictions (for EDA)

    Parameters
    ----------
    all_curr_matchups : list
        DataFrames of matchups from each round
    all_curr_rounds : list
        DataFrames of game results from each round

    Returns
    -------
    bracket_preds : DataFrame
        Complete, cleaned bracket predictions DataFrame
    """
    # Concatenate all tournament data into their respective DataFrames
    all_matchups = pd.concat(all_curr_matchups, ignore_index=True)
    bracket_preds = pd.concat(all_curr_rounds, ignore_index=True)

    # Concatenate all relevant tournament data into a single DataFrame
    bracket_preds = pd.concat(
        [all_matchups['Seed'], bracket_preds['Team_Favorite'], all_matchups['Seed.1'], bracket_preds.iloc[:, 1:]], 
        axis=1
    )

    # Rename columns for greater interpretability
    bracket_preds.rename(columns = {
        'Seed': 'Seed_Favorite',
        'Seed.1': 'Seed_Underdog',
    }, inplace=True)

    # Change round numbers to round names (i.e. Sweet 16, Elite 8, etc.)
    bidirectional_rounds_str_numeric(bracket_preds)

    return bracket_preds