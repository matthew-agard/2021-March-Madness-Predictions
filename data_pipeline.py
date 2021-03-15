import pandas as pd
from data_fetch import get_team_data, get_rankings_data, get_coach_data, get_current_bracket
from data_clean import clean_basic_stats, clean_adv_stats, clean_coach_stats, reclean_all_season_stats, clean_tourney_data, clean_curr_round_data, fill_playin_teams
from data_merge import merge_clean_team_stats, merge_clean_rankings, merge_clean_coaches, merge_clean_tourney_games
from feature_engineering import team_points_differentials, rounds_to_numeric, matchups_to_underdog_relative, scale_features, create_bracket_round
from model_evaluation import probs_to_preds


def regular_season_stats(year):
    # Fetch & clean basic regular season stats
    season_basic_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html",
                                    attrs={'id': 'basic_school_stats'})
    clean_season_basic_df = clean_basic_stats(season_basic_df)
    
    # Fetch & clean advanced regular season stats
    season_adv_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html", 
                                attrs={'id': 'adv_school_stats'})
    clean_season_adv_df = clean_adv_stats(season_adv_df)

    # Merge all cleaned regular season stats
    return clean_season_basic_df, merge_clean_team_stats(clean_season_basic_df, clean_season_adv_df)


def team_rankings(year, season_stats):
    # Fetch team rankings data (already cleaned)
    rankings_df = get_rankings_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-ratings.html")

    # Merge rankings data to all team stats
    return merge_clean_rankings(season_stats, rankings_df)


def coach_performance(year, stats_rankings):
    # Fetch & clean coach performance data
    coaches_df = get_coach_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-coaches.html")
    clean_coaches_df = clean_coach_stats(coaches_df)
    
    # Merge coach data to all regular season data
    return merge_clean_coaches(stats_rankings, clean_coaches_df)


def all_team_season_data(year):
    clean_season_basic_df, team_season_stats_df = regular_season_stats(year)
    team_stats_rankings_df = team_rankings(year, team_season_stats_df)
    all_season_stats_df = coach_performance(year, team_stats_rankings_df)

    return all_season_stats_df, clean_season_basic_df


def hist_tournament_games(year, all_stats, basic_stats):
    # Reclean all team names & season stats (pre-tourney merge)
    clean_all_season_stats_df = reclean_all_season_stats(year, all_stats, basic_stats)
    
    # Fetch tournament game data
    mm_games_df = get_team_data(url=("https://apps.washingtonpost.com/sports/search/?pri_school_id=&pri_conference=&pri_coach"
                                "=&pri_seed_from=1&pri_seed_to=16&pri_power_conference=&pri_bid_type=&opp_school_id"
                                "=&opp_conference=&opp_coach=&opp_seed_from=1&opp_seed_to=16&opp_power_conference"
                                f"=&opp_bid_type=&game_type=7&from={year}&to={year}&submit="), 
                                attrs={'class': 'search-results'}, header=0)
    
    # Clean & merge regular season data to tournament games
    if not mm_games_df.empty:
        clean_mm_df = clean_tourney_data(year, mm_games_df, clean_all_season_stats_df)
        mm_data_df = merge_clean_tourney_games(clean_mm_df, clean_all_season_stats_df)
    else:
        mm_data_df = pd.DataFrame()

    return mm_data_df


def dataset_pipeline(years):
    all_mm_data_df = pd.DataFrame()

    for year in years:    
        all_season_stats_df, clean_season_basic_df = all_team_season_data(year)
        year_mm_data_df = hist_tournament_games(year, all_season_stats_df, clean_season_basic_df)

        all_mm_data_df = pd.concat([all_mm_data_df, year_mm_data_df], ignore_index=True)

    return all_mm_data_df


def feature_pipeline(df):
    try:
        rounds_to_numeric(df)
    except KeyError:
        pass

    team_points_differentials(df)
    matchups_to_underdog_relative(df)
    finalized_df = scale_features(df)

    return finalized_df


def round_pipeline(year, curr_round, all_curr_matchups, clean_curr_season_data, null_drops):
    # Generate non-engineered round
    if curr_round not in [0, 1]:
        generated_round = create_bracket_round(all_curr_matchups[curr_round-1])
    else:
        generated_round = all_curr_matchups[curr_round]

    # Convert round matchups to favorite-underdog format
    cleaned_generated_round = clean_tourney_data(year, generated_round, clean_curr_season_data)

    # Merge all team season data to teams in matchups
    all_round_data = merge_clean_tourney_games(cleaned_generated_round, clean_curr_season_data)

    # Prepare df for prediction via feature pipeline preprocessing
    schools = ['Team_Favorite', 'Team_Underdog']
    school_matchups_df = all_round_data[schools]

    all_round_data.drop(schools + null_drops, axis=1, inplace=True)

    school_matchups_df['Round'] = [curr_round] * len(school_matchups_df)

    curr_X = feature_pipeline(all_round_data)

    return all_round_data, curr_X, school_matchups_df


def bracket_pipeline(year, play_in, first_round, model, thresh, null_drops):
    all_curr_season_data, curr_season_basic_df = all_team_season_data(year)
    clean_curr_season_data = reclean_all_season_stats(year, all_curr_season_data, curr_season_basic_df)

    all_curr_matchups = [play_in, first_round]
    all_curr_rounds = [play_in, first_round]

    for curr_round in range(7):    
        all_round_data, curr_X, school_matchups_df = round_pipeline(year, curr_round, all_curr_matchups, 
                                                                    clean_curr_season_data, null_drops)
        # Make & store predictions
        curr_y_probs = model.predict_proba(curr_X)[:, 1]
        school_matchups_df['Underdog_Upset'] = probs_to_preds(curr_y_probs, thresh)
        
        # Clean current predictions for use in next round
        curr_X, school_matchups_df = clean_curr_round_data(all_round_data, curr_X, school_matchups_df)
        
        # Store necessary data        
        if curr_round in [0, 1]:
            all_curr_matchups[curr_round] = curr_X
            all_curr_rounds[curr_round] = school_matchups_df
        else:
            all_curr_matchups.append(curr_X)
            all_curr_rounds.append(school_matchups_df)

        # Fill first round nulls with play-in winners (when necesssary)
        if curr_round == 0:
            fill_playin_teams(all_curr_matchups)

    return all_curr_rounds, all_curr_matchups