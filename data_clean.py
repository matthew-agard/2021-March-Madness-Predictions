import pandas as pd
import numpy as np
from data_integrity import coach_to_season_integrity_dict, season_to_tourney_integrity_dict
from feature_engineering import totals_to_game_average, create_faves_underdogs, create_target_variable
from datetime import datetime
current_year = datetime.now().year


def clean_basic_stats(year, df):
    useless_feats = ['Rk', 'MP'] + [col for col in df.columns 
                                    if ('Unnamed' in col) or ('W.' in col) or ('L.' in col)]
    lin_dep_feats = ['W', 'L', 'SRS', 'FGA', '3PA', 'FTA']
    feat_drops = useless_feats + lin_dep_feats
    
    df.drop(feat_drops, axis=1, inplace=True)
    df = df[(df['School'] != 'School') & (df['G'] != 'Overall')]

    # ADDRESS LATER FOR create_dataset([2021])
    ncaa_df = df[df['School'].str.contains('NCAA')] if (year != current_year) else df

    return ncaa_df


def clean_adv_stats(df):
    return pd.concat([df['School'], df.iloc[:, -13:]], axis=1)


def clean_coach_stats(coach_df):
    coach_df['Coach_Team'].replace(coach_to_season_integrity_dict, inplace=True)

    coach_df.iloc[:, 1:] = coach_df.iloc[:, 1:].replace('', '0')

    return coach_df


def reclean_all_season_stats(all_season_df, season_basic_df):
    all_season_df['School'].replace(season_to_tourney_integrity_dict, inplace=True)
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
            mm_df[mm_df_struct[j] + "_" + key] = faves_unds[key][:, j]

    if year != current_year:
        mm_df['Underdog_Upset'] = create_target_variable(mm_df)
            
    mm_df_drop = ['Seed', 'Team'] + [col for col in mm_df.columns if ('.1' in col)]
    mm_df.drop(mm_df_drop, axis=1, inplace=True)

    return mm_df
    

def feature_null_counts(df):
    nulls = df.isnull().sum().sort_values(ascending=False)
    return nulls[nulls > 0]


def get_null_rows(null_fills, df):
    rows = df[df[null_fills].isnull().any(axis=1)]
    return rows[['Year'] + null_fills]