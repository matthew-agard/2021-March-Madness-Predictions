import pandas as pd
import numpy as np


def teams_to_faves_underdogs(mm_df, season_df):
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