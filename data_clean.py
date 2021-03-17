import pandas as pd
import numpy as np
from datetime import datetime
from data_integrity import coach_to_season_dict, hist_season_to_tourney_dict, curr_season_to_tourney_dict
from feature_engineering import totals_to_game_average, create_faves_underdogs, bidirectional_rounds_numeric, create_target_variable

current_year = datetime.now().year


def clean_basic_stats(df):
    useless_feats = ['Rk', 'MP'] + [col for col in df.columns 
                                    if ('Unnamed' in col) or ('W.' in col) or ('L.' in col)]
    lin_dep_feats = ['W', 'L', 'SRS', 'FGA', '3PA', 'FTA']    
    df.drop(useless_feats + lin_dep_feats, axis=1, inplace=True)

    df = df[(df['School'] != 'School') & (df['G'] != 'Overall')]
    ncaa_df = df[df['School'].str.contains('NCAA')]

    return ncaa_df


def clean_adv_stats(df):
    return pd.concat([df['School'], df.iloc[:, -13:]], axis=1)


def clean_coach_stats(coach_df):
    coach_df['Coach_Team'].replace(coach_to_season_dict, inplace=True)

    coach_df.iloc[:, 1:] = coach_df.iloc[:, 1:].replace('', '0')

    return coach_df


def reclean_all_season_stats(year, all_season_df, season_basic_df):
    if year != current_year:
        all_season_df['School'].replace(hist_season_to_tourney_dict, inplace=True)
    else:
        all_season_df['School'].replace(curr_season_to_tourney_dict, inplace=True)

    totals_to_game_average(all_season_df, season_basic_df)

    return all_season_df


def clean_tourney_data(year, mm_df, season_df):
    if year != current_year:
        for col in ['Round', 'Team', 'Team.1']:
            mm_df[col] = mm_df[col].apply(lambda name: name[:-(len(name) // 2)].strip())

    faves_unds = create_faves_underdogs(mm_df, season_df)
    mm_df_struct = ['Seed', 'Team', 'Score']

    for key in faves_unds.keys():    
        for j in range(len(mm_df_struct)):
            try:
                mm_df[mm_df_struct[j] + "_" + key] = faves_unds[key][:, j]
            except IndexError:
                continue
    try:
        mm_df['Underdog_Upset'] = create_target_variable(mm_df)
    except KeyError:
        pass
            
    mm_df_drop = ['Seed', 'Team', 'Seed.1', 'Team.1']
    mm_df.drop(mm_df_drop, axis=1, inplace=True)

    return mm_df
    

def feature_null_counts(df):
    nulls = df.isnull().sum().sort_values(ascending=False)
    return nulls[nulls > 0]


def get_null_rows(null_fills, df):
    rows = df[df[null_fills].isnull().any(axis=1)]
    return rows[['Year'] + null_fills]


def clean_round_cols(df):
    df.rename(columns = {
        'Seed_Favorite': 'Seed',
        'Team_Favorite': 'Team',
        'Seed_Underdog': 'Seed.1',
        'Team_Underdog': 'Team.1',
    }, inplace=True)


def clean_curr_round_data(all_round_data, curr_X, school_matchups_df):
    curr_X[['Seed_Favorite', 'Seed_Underdog']] = all_round_data[['Seed_Favorite', 'Seed_Underdog']].astype(float).astype(int)
    curr_X = pd.concat([school_matchups_df, curr_X], axis=1)

    clean_round_cols(curr_X)
    curr_X.drop_duplicates(subset=['Team', 'Team.1'], inplace=True)
    curr_X.index = range(len(curr_X))
    
    school_matchups_df.drop_duplicates(subset=['Team_Favorite', 'Team_Underdog'], inplace=True)
    school_matchups_df.index = range(len(school_matchups_df))

    return curr_X, school_matchups_df


def fill_playin_teams(all_curr_matchups):
    for index, data in all_curr_matchups[0].iterrows():
        winner_seed = data['Seed'] if (data['Underdog_Upset'] == 0) else data['Seed.1']
        winner_team = all_curr_matchups[0].iloc[index, data['Underdog_Upset']]

        all_curr_matchups[1]['Seed.1'].fillna(winner_seed, inplace=True, limit=1)
        all_curr_matchups[1]['Team.1'].fillna(winner_team, inplace=True, limit=1)

    all_curr_matchups[0][['Seed', 'Seed.1']] = all_curr_matchups[0][['Seed', 'Seed.1']].astype(int)
    all_curr_matchups[1][['Seed', 'Seed.1']] = all_curr_matchups[1][['Seed', 'Seed.1']].astype(int)


def clean_bracket(all_curr_matchups, all_curr_rounds):
    all_matchups = pd.concat(all_curr_matchups, ignore_index=True)
    bracket_preds = pd.concat(all_curr_rounds, ignore_index=True)

    bracket_preds = pd.concat([all_matchups[['Seed', 'Seed.1']], bracket_preds], axis=1)
    bracket_preds.rename(columns = {
        'Seed': 'Seed_Favorite',
        'Seed.1': 'Seed_Underdog',
    }, inplace=True)
    bidirectional_rounds_numeric(bracket_preds)

    return bracket_preds