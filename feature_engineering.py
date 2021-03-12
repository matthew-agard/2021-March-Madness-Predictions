import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def totals_to_game_average(all_season_df, season_basic_df):
    for i in range(1, len(all_season_df.columns)):
        all_season_df.iloc[:, i] = all_season_df.iloc[:, i].astype(float)
        
    for col in season_basic_df.columns:
        if (col not in ['School', 'G', 'SOS']) and ('%' not in col):
            all_season_df[col + "/Game"] = np.round(all_season_df[col] / all_season_df['G'], 1)
            all_season_df.drop(col, axis=1, inplace=True)
            
    all_season_df.drop('G', axis=1, inplace=True)


def create_faves_underdogs(mm_df, season_df):
    faves, underdogs = [], []

    for index, data in mm_df.iterrows():
        team_arr = [data['Seed'], data['Team'], data['Score']]
        team1_arr = [data['Seed.1'], data['Team.1'], data['Score.1']]

        if data['Seed.1'] == data['Seed']:
            team_win_pct = float(season_df[season_df['School'] == data['Team']]['W-L%'])
            team1_win_pct = float(season_df[season_df['School'] == data['Team.1']]['W-L%'])
            
            if team_win_pct > team1_win_pct:
                underdogs.append(team1_arr)
                faves.append(team_arr)
            else:
                underdogs.append(team_arr)
                faves.append(team1_arr)
            
        elif data['Seed.1'] > data['Seed']:
            underdogs.append(team1_arr)
            faves.append(team_arr)

        else:
            underdogs.append(team_arr)
            faves.append(team1_arr)

    faves_unds = {
    'Favorite': np.array(faves),
    'Underdog': np.array(underdogs)
    }

    return faves_unds


def team_points_differentials(df):
    for team in ['Favorite', 'Underdog']:
        df['PtsDiff_' + team] = df['Tm./Game_' + team] - df['Opp./Game_' + team]
        df.drop(['Tm./Game_' + team, 'Opp./Game_' + team], axis=1, inplace=True)


def rounds_to_numeric(df):
    df['Round'].replace({
        'Play-In': 0,
        'First Round': 1,
        'Second Round': 2,
        'Sweet 16': 3,
        'Elite Eight': 4,
        'Final Four': 5,
        'National Championship': 6
    }, 
    inplace=True)


def matchups_to_underdog_relative(df):
    team_stat_cols = set([col.replace('_Underdog', '').replace('_Favorite', '') for col in df.columns])
    non_relative_cols = ['Round', 'Seed', 'Underdog_Upset']

    for col in team_stat_cols:    
        if col not in non_relative_cols:
            df['Underdog_Rel_' + col] = df[col + '_Underdog'] - df[col + '_Favorite']
            df.drop([col + '_Underdog', col + '_Favorite'], axis=1, inplace=True)


def scale_features(df):
    scaler = StandardScaler()

    try:
        rescale = scaler.fit_transform(df.drop('Underdog_Upset', axis=1))
        df_scaled = pd.DataFrame(rescale, index=df.index, 
                                columns=[col for col in df.columns if col != 'Underdog_Upset'])
        full_df = pd.concat([df_scaled, df['Underdog_Upset']], axis=1)
    except KeyError:
        rescale = scaler.fit_transform(df)
        full_df = pd.DataFrame(rescale, index=df.index, columns=df.columns)
    
    return full_df


def create_next_bracket_round(prev_round):
    matchups = prev_round[['Team', 'Team.1']]
    winners = []

    for index, data in prev_round.iterrows():
        winner_seed = data['Seed'] if (data['Underdog_Upset'] == 0) else data['Seed.1']
        winner_team = matchups.iloc[index, data['Underdog_Upset']]

        winners.append(tuple((winner_seed, winner_team)))

    winners = np.array(winners).reshape((len(winners) // 2), 4)

    return winners

 
def create_target_variable(mm_df):
    target = (mm_df['Score_Underdog'] > mm_df['Score_Favorite']).astype(int)
    
    score_cols = [col for col in mm_df.columns if ('Score' in col)]
    mm_df.drop(score_cols, axis=1, inplace=True)

    return target