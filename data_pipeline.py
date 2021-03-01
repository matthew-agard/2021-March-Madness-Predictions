import pandas as pd
from data_fetch import get_team_data, get_rankings_data, get_coach_data
from data_clean import clean_basic_stats, clean_adv_stats, clean_coach_stats, reclean_all_season_stats, clean_tourney_data
from data_merge import merge_clean_team_stats, merge_clean_rankings, merge_clean_coaches, merge_clean_tourney_games


def create_dataset(years):
    all_mm_data_df = pd.DataFrame()

    for year in years:    
        """Team Regular Season Stats"""
        # Fetch & clean basic regular season stats
        season_basic_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html",
                                        attrs={'id': 'basic_school_stats'})
        clean_season_basic_df = clean_basic_stats(season_basic_df)
        
        # Fetch & clean advanced regular season stats
        season_adv_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html", 
                                    attrs={'id': 'adv_school_stats'})
        clean_season_adv_df = clean_adv_stats(season_adv_df)

        # Merge all cleaned regular season stats
        team_season_stats_df = merge_clean_team_stats(clean_season_basic_df, clean_season_adv_df)
        
        
        """Team Rankings"""
        # Fetch team rankings data (already cleaned)
        rankings_df = get_rankings_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-ratings.html")

        # Merge rankings data to all team stats
        team_stats_rankings_df = merge_clean_rankings(team_season_stats_df, rankings_df)   

        
        """Coach Tournament Performance"""
        # Fetch & clean coach performance data
        coaches_df = get_coach_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-coaches.html")
        clean_coaches_df = clean_coach_stats(coaches_df)
        
        # Merge coach data to all regular season data
        all_season_stats_df = merge_clean_coaches(team_stats_rankings_df, clean_coaches_df)
    

        """Tournament Game Data"""
        # Reclean all team names & season stats (pre-tourney merge)
        clean_all_season_stats_df = reclean_all_season_stats(all_season_stats_df, clean_season_basic_df)
        
        # Fetch & clean tournament game data
        mm_games_df = get_team_data(url=("https://apps.washingtonpost.com/sports/search/?pri_school_id=&pri_conference=&pri_coach"
                                    "=&pri_seed_from=1&pri_seed_to=16&pri_power_conference=&pri_bid_type=&opp_school_id"
                                    "=&opp_conference=&opp_coach=&opp_seed_from=1&opp_seed_to=16&opp_power_conference"
                                    f"=&opp_bid_type=&game_type=7&from={year}&to={year}&submit="), 
                                    attrs={'class': 'search-results'}, header=0)    
        clean_mm_df = clean_tourney_data(mm_games_df, clean_all_season_stats_df)
            
        # Merge regular season data to tournament games
        year_mm_data_df = merge_clean_tourney_games(clean_mm_df, clean_all_season_stats_df)
        
        all_mm_data_df = pd.concat([all_mm_data_df, year_mm_data_df], ignore_index=True)

    return all_mm_data_df