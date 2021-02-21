import pandas as pd
import numpy as np


def totals_to_game_average(all_season_df, season_basic_df):
    for i in range(1, len(all_season_df.columns)):
        all_season_df.iloc[:, i] = all_season_df.iloc[:, i].astype(float)
        
    for col in season_basic_df.columns:
        if (col not in ['School', 'G', 'SOS']) and ('%' not in col):
            all_season_df[col + "/Game"] = np.round(all_season_df[col] / all_season_df['G'], 1)
            all_season_df.drop(col, axis=1, inplace=True)
            
    all_season_df.drop('G', axis=1, inplace=True)

    return all_season_df


def create_faves_underdogs(mm_df, season_df):
    faves, underdogs = [], []

    for index, data in mm_df.iterrows():
        if data['Seed.1'] == data['Seed']:
            team_win_pct = float(season_df[season_df['School'] == data['Team']]['W-L%'])
            team1_win_pct = float(season_df[season_df['School'] == data['Team.1']]['W-L%'])
            
            if team_win_pct > team1_win_pct:
                underdogs.append([data['Seed.1'], data['Team.1'], data['Score.1']])
                faves.append([data['Seed'], data['Team'], data['Score']])
            else:
                underdogs.append([data['Seed'], data['Team'], data['Score']])
                faves.append([data['Seed.1'], data['Team.1'], data['Score.1']])
            
        elif data['Seed.1'] > data['Seed']:
            underdogs.append([data['Seed.1'], data['Team.1'], data['Score.1']])
            faves.append([data['Seed'], data['Team'], data['Score']])

        else:
            underdogs.append([data['Seed'], data['Team'], data['Score']])
            faves.append([data['Seed.1'], data['Team.1'], data['Score.1']])

    faves_unds = {
    'Favorite': np.array(faves),
    'Underdog': np.array(underdogs)
    }

    return faves_unds

 
def create_target_variable(mm_df):
    return (mm_df['Score_Underdog'] > mm_df['Score_Favorite']).astype(int)