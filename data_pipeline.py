import pandas as pd
from data_fetch import get_team_data, get_rankings_data, get_coach_data, get_current_bracket
from data_clean import clean_basic_stats, clean_adv_stats, clean_coach_stats, reclean_all_season_stats, clean_tourney_data
from data_merge import merge_clean_team_stats, merge_clean_rankings, merge_clean_coaches, merge_clean_tourney_games
from feature_engineering import team_points_differentials, rounds_to_numeric, matchups_to_underdog_relative, scale_features
from datetime import datetime
current_year = datetime.now().year


def regular_season_stats(year):
    # Fetch & clean basic regular season stats
    season_basic_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html",
                                    attrs={'id': 'basic_school_stats'})
    clean_season_basic_df = clean_basic_stats(year, season_basic_df)
    
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


def tournament_games(year, all_stats, basic_stats):
    # Reclean all team names & season stats (pre-tourney merge)
    clean_all_season_stats_df = reclean_all_season_stats(all_stats, basic_stats)
    
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
        year_mm_data_df = tournament_games(year, all_season_stats_df, clean_season_basic_df)

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