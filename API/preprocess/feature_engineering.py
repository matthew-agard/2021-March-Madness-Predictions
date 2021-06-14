"""Feature Engineering Helper Functions

This script is used as a helper module in the data_clean and data_pipeline scripts.

The following functions are present:
    * totals_to_game_average
    * create_faves_underdogs
    * team_points_differentials
    * bidirectional_rounds_str_numeric
    * matchups_to_underdog_relative
    * scale_features
    * create_bracket_round
    * create_bracket_winners
    * create_target_variable

Requires a minimum of the 'pandas', 'numpy', and 'sklearn' libraries, as well as 
the 'data_integrity' helper module, being present in your environment to run.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_integrity import rounds_str_to_numeric, rounds_numeric_to_str


def totals_to_game_average(all_season_df, season_basic_df):
    """Convert team season stats totals into per game averages

    Parameters
    ----------
    all_season_df : DataFrame
        Complete regular season dataset (uncleaned)
    season_basic_df : DataFrame
        Cleaned basic regular season team stats (only column names used here)
    """
    # Convert all numeric datatypes to from str to float
    for i in range(1, len(all_season_df.columns)):
        all_season_df.iloc[:, i] = all_season_df.iloc[:, i].astype(float)
    
    # Convert all regular & advanced team stats from season totals to per game averages
    for col in season_basic_df.columns:
        if (col not in ['School', 'G', 'SOS']) and ('%' not in col):
            all_season_df[col + "/Game"] = np.round(all_season_df[col] / all_season_df['G'], 1)
            # Drop season total features
            all_season_df.drop(col, axis=1, inplace=True)


def create_faves_underdogs(mm_df, season_df):
    """Convert team listings into favorites-underdogs matchups

    Parameters
    ----------
    mm_df : DataFrame
        Freshly scraped tournament matchup data
    season_df : DataFrame
        Cleaned basic regular season team stats

    Returns
    -------
    faves_unds : dict
        Two arrays, one containing team matchup data for favorites and the other for underdogs
    """
    faves, underdogs = [], []

    for index, data in mm_df.iterrows():
        # Get team matchup data
        # Seed --> team_arr, Seed.1 --> team1_arr
        try:
            team_arr = [data['Seed'], data['Team'], data['Score']]
            team1_arr = [data['Seed.1'], data['Team.1'], data['Score.1']]
        # Catch key error for missing scores (when creating tournament matchups outside of dataset)
        except KeyError:
            team_arr = [data['Seed'], data['Team']]
            team1_arr = [data['Seed.1'], data['Team.1']]

        # In this case, the team linked to Seed is the underdog; populate corresponding arrays
        if data['Seed.1'] < data['Seed']:
            underdogs.append(team_arr)
            faves.append(team1_arr)

        # In this case, the team linked to Seed.1 is the underdog; populate corresponding arrays
        elif data['Seed.1'] > data['Seed']:
            underdogs.append(team1_arr)
            faves.append(team_arr)

        # Seeds are equivalent
        else:
            # Get regular season win percentage for both teams
            team_win_pct = float(season_df[season_df['School'] == data['Team']]['W-L%'])
            team1_win_pct = float(season_df[season_df['School'] == data['Team.1']]['W-L%'])
            
            # Whoever has the better record is the favorite, else they're the underdog; populate corresponding arrays
            if team_win_pct > team1_win_pct:
                underdogs.append(team1_arr)
                faves.append(team_arr)
            else:
                underdogs.append(team_arr)
                faves.append(team1_arr)

    # Return favorite-underdogs arrays as a single dictionary, referenced by their corresponding key
    faves_unds = {
        'Favorite': np.array(faves),
        'Underdog': np.array(underdogs)
    }

    return faves_unds


def team_points_differentials(df):
    """Convert team points/game features into point differential feature

    Parameters
    ----------
    df : DataFrame
        Fully merged and cleaned tournament data
    """
    # Create points differential feature from existing per game features
    for team in ['Favorite', 'Underdog']:
        df['PtsDiff_' + team] = df['Tm./Game_' + team] - df['Opp./Game_' + team]
        # Remove old points/game features to avoid linear dependency
        df.drop(['Tm./Game_' + team, 'Opp./Game_' + team], axis=1, inplace=True)


def bidirectional_rounds_str_numeric(df):
    """Convert rounds to integers for modeling, then back to strings for EDA

    Parameters
    ----------
    df : DataFrame
        Fully merged and cleaned tournament data
    """
    # Determine which replacement dictionary to use based on datatype, then apply said replacements
    rounds_replace = rounds_str_to_numeric if (df['Round'].dtype == object) else rounds_numeric_to_str
    df['Round'].replace(rounds_replace, inplace=True)


def matchups_to_underdog_relative(df):
    """Convert favorite-underdog features to a single class of 'Underdog_Relative' features

    Parameters
    ----------
    df : DataFrame
        Fully merged and cleaned tournament data
    """
    # Get set of all features that should be made relative
    team_stat_cols = set([col.replace('_Underdog', '').replace('_Favorite', '') for col in df.columns])
    # Exclude round, seed, and target variable from this process
    team_stat_cols.difference_update(['Round', 'Seed', 'Underdog_Upset'])

    # Perform feature conversion
    for col in team_stat_cols:    
        df['Underdog_Rel_' + col] = df[col + '_Underdog'] - df[col + '_Favorite']
        # Remove old favorite-underdog features to avoid linear dependency
        df.drop([col + '_Underdog', col + '_Favorite'], axis=1, inplace=True)


def scale_features(df):
    """'Center the data' of all numerical features; levels playing field for feature importances

    Parameters
    ----------
    df : DataFrame
        Fully merged and cleaned tournament data

    Returns
    -------
    full_df : DataFrame
        Fully merged and cleaned tournament data that has been scaled
    """
    # import StandardScaler object
    scaler = StandardScaler()

    # Rescale data, then format it according to the structure of the original DataFrame
    try:
        rescale = scaler.fit_transform(df.drop('Underdog_Upset', axis=1))
        df_scaled = pd.DataFrame(rescale, index=df.index, 
                                columns=[col for col in df.columns if col != 'Underdog_Upset'])
        full_df = pd.concat([df_scaled, df['Underdog_Upset']], axis=1)
    # Catch key error when scaling data that doesn't contain a target variable
    except KeyError:
        rescale = scaler.fit_transform(df)
        full_df = pd.DataFrame(rescale, index=df.index, columns=df.columns)
    
    return full_df


def create_bracket_round(prev_round):
    """Generate matchups of a subsequent round based on a previous round's outcomes

    Parameters
    ----------
    prev_round : DataFrame
        Round whose outcome data is used for generating subsequent round matchups

    Returns
    -------
    next_round : DataFrame
        Round generated from winners of 'prev_round'
    """
    # Get teams for each matchups
    matchups = prev_round[['Team', 'Team.1']]
    winners = []

    # Get the team data (seed & team name) from each 'prev_round' matchup
    for index, data in prev_round.iterrows():
        winner_seed = data['Seed'] if (data['Underdog_Upset'] == 0) else data['Seed.1']
        # If 'Underdog_Upset' == 0 then 'Team' is the winner, else 'Team.1' is the winner
        winner_team = matchups.iloc[index, data['Underdog_Upset']]
        # Append a single winner to the initialized list
        winners.append([winner_seed, winner_team])

    # Reshape the n winners into n/2 matchups for a DataFrame of the subsequent round
    winners = np.array(winners).reshape((len(winners) // 2), 4)
    next_round = pd.DataFrame(winners, columns=['Seed', 'Team', 'Seed.1', 'Team.1'])

    return next_round


def create_bracket_winners(bracket):
    """Clearly denote the winners of each matchup for the entire generated bracket

    Parameters
    ----------
    bracket : DataFrame
        Generated bracket predictions for current year's tournament
    """
    winners = []

    # Determine which team won, and append their name to the above list
    for index, data in bracket.iterrows():
        team_winner = data['Team_Underdog'] if (data['Underdog_Upset'] == 1) else data['Team_Favorite']
        winners.append(team_winner)
        
    # Convert the list into a Series on the 'bracket' DataFrame
    bracket['Winner'] = winners

 
def create_target_variable(mm_df):
    """Engineer historical dataset target variable

    Parameters
    ----------
    mm_df : DataFrame
        Freshly scraped tournament matchup data

    Returns
    -------
    target : Series
        Historical tournament matchups outcomes
    """
    # Create binary target variable denoting whether or not the underdog won the game
    target = (mm_df['Score_Underdog'] > mm_df['Score_Favorite']).astype(int)
    
    # Drop score features to avoid data leakage
    score_cols = [col for col in mm_df.columns if ('Score' in col)]
    mm_df.drop(score_cols, axis=1, inplace=True)

    return target