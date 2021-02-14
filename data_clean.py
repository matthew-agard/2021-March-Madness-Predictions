import pandas as pd
import numpy as np
from data_integrity import coach_to_season_integrity_dict, season_to_tourney_integrity_dict
from feature_engineering import teams_to_faves_underdogs


def clean_basic_stats(df):
    useless_feats = ['Rk', 'MP'] + [col for col in df.columns 
                                    if ('Unnamed' in col) or ('W.' in col) or ('L.' in col)]
    lin_dep_feats = ['W', 'L', 'SRS', 'FGA', '3PA', 'FTA']
    feat_drops = useless_feats + lin_dep_feats
    
    df.drop(feat_drops, axis=1, inplace=True)
    df.dropna(inplace=True)

    ncaa_df = df[df['School'].str.contains('NCAA')]

    return ncaa_df


def clean_adv_stats(df):
    adv_df = pd.concat([df['School'], df.iloc[:, -13:]], axis=1)
    
    return adv_df


def clean_coach_stats(coach_df):
    coach_df['Coach_Team'].replace(coach_to_season_integrity_dict, inplace=True)
    coach_df.iloc[:, 1:] = coach_df.iloc[:, 1:].replace('', '0')

    return coach_df


def clean_all_stats(all_season_df, season_basic_df):
    for i in range(1, len(all_season_df.columns)):
        all_season_df.iloc[:, i] = all_season_df.iloc[:, i].astype(float)
        
    for col in season_basic_df.columns:
        if (col not in ['School', 'G', 'SOS']) and ('%' not in col):
            all_season_df[col + "/Game"] = np.round(all_season_df[col] / all_season_df['G'], 1)
            all_season_df.drop(col, axis=1, inplace=True)
            
    all_season_df.drop('G', axis=1, inplace=True)

    all_season_df['School'].replace(season_to_tourney_integrity_dict, inplace=True)

    return all_season_df


def clean_tourney_data(mm_df, season_df):
    for col in ['Round', 'Team', 'Team.1']:
        mm_df[col] = mm_df[col].apply(lambda name: name[:-(len(name) // 2)].strip())

    faves_unds = teams_to_faves_underdogs(mm_df, season_df)
    mm_df_struct = ['Seed', 'Team', 'Score']

    for key in faves_unds.keys():    
        for j in range(len(mm_df_struct)):
            mm_df[mm_df_struct[j] + "_" + key] = faves_unds[key][:, j]
            
    mm_df['Underdog_Upset'] = (mm_df['Score_Underdog'] > mm_df['Score_Favorite']).astype(int)
            
    mm_df_drop = mm_df_struct + [col for col in mm_df.columns if 
                                ('.1' in col) or ('Score' in col) or ('Year' in col)]

    mm_df.drop(mm_df_drop, axis=1, inplace=True)

    return mm_df