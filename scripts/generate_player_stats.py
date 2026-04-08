import pandas as pd
import numpy as np
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "raw_data"
PROCESSED_DATA_DIR = REPO_ROOT / "data"
BATTING_STATS_PATH = PROCESSED_DATA_DIR / "batter_stats.csv"
BOWLING_STATS_PATH = PROCESSED_DATA_DIR / "bowler_stats.csv"
MASTER_STATS_PATH = PROCESSED_DATA_DIR / "master_player_stats.csv"

def load_all_data(format_type="t20s"):
    data_dir = RAW_DATA_DIR / format_type
    all_files = [str(data_dir / f) for f in os.listdir(data_dir) 
                 if f.endswith('.csv') and not f.endswith('_info.csv')]
    
    print(f"Loading {len(all_files)} match files... (This might take a minute)")
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    return df

def generate_comprehensive_stats():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_data("t20s")
    
    # ---------------------------------------------------------
    # 1. PRE-PROCESSING & COLUMN CLEANING
    # Cricsheet might not create columns for extras if none occurred in a match subset,
    # so we ensure they exist and fill NaNs with 0.
    # ---------------------------------------------------------
    extra_cols = ['wides', 'noballs', 'byes', 'legbyes', 'penalty']
    for col in extra_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
            
    if 'wicket_type' not in df.columns:
        df['wicket_type'] = np.nan
    if 'player_dismissed' not in df.columns:
        df['player_dismissed'] = np.nan

    # ---------------------------------------------------------
    # 2. BATTER STATISTICS
    # ---------------------------------------------------------
    print("Calculating Batter Statistics...")
    
    # A ball is only 'faced' if it's not a wide
    df['is_ball_faced'] = np.where(df['wides'] > 0, 0, 1)
    
    bat_group = df.groupby('striker').agg(
        innings_batted=('match_id', 'nunique'),
        runs_scored=('runs_off_bat', 'sum'),
        balls_faced=('is_ball_faced', 'sum'),
        dots=('runs_off_bat', lambda x: (x == 0).sum()),
        ones=('runs_off_bat', lambda x: (x == 1).sum()),
        twos=('runs_off_bat', lambda x: (x == 2).sum()),
        threes=('runs_off_bat', lambda x: (x == 3).sum()),
        fours=('runs_off_bat', lambda x: (x == 4).sum()),
        sixes=('runs_off_bat', lambda x: (x == 6).sum())
    ).reset_index()

    # Calculate dismissals (where the player_dismissed is the striker)
    dismissals = df[df['player_dismissed'] == df['striker']].groupby('striker').size().reset_index(name='times_dismissed')
    bat_stats = pd.merge(bat_group, dismissals, on='striker', how='left')
    bat_stats['times_dismissed'] = bat_stats['times_dismissed'].fillna(0)

    # Derived Stats
    bat_stats['batting_average'] = np.where(bat_stats['times_dismissed'] > 0, 
                                            bat_stats['runs_scored'] / bat_stats['times_dismissed'], 
                                            bat_stats['runs_scored']) # If never dismissed, avg is total runs
    bat_stats['strike_rate'] = np.where(bat_stats['balls_faced'] > 0, 
                                        (bat_stats['runs_scored'] / bat_stats['balls_faced']) * 100, 
                                        0)
    bat_stats['boundary_pct'] = np.where(bat_stats['balls_faced'] > 0, 
                                         (bat_stats['fours'] + bat_stats['sixes']) / bat_stats['balls_faced'], 
                                         0)

    # ---------------------------------------------------------
    # 3. BOWLER STATISTICS
    # ---------------------------------------------------------
    print("Calculating Bowler Statistics...")
    
    # Legal deliveries: not a wide, not a no-ball
    df['is_legal_delivery'] = np.where((df['wides'] > 0) | (df['noballs'] > 0), 0, 1)
    
    # Bowler runs: runs off bat + wides + noballs (excludes byes/legbyes)
    df['bowler_runs_conceded'] = df['runs_off_bat'] + df['wides'] + df['noballs']
    
    # Bowler wickets: dismissals excluding run outs, retired hurts, obstructing field
    non_bowler_wickets = ['run out', 'retired hurt', 'obstructing the field', 'retired out']
    df['is_bowler_wicket'] = np.where(df['wicket_type'].notna() & ~df['wicket_type'].isin(non_bowler_wickets), 1, 0)

    bowl_group = df.groupby('bowler').agg(
        innings_bowled=('match_id', 'nunique'),
        legal_balls_bowled=('is_legal_delivery', 'sum'),
        runs_conceded=('bowler_runs_conceded', 'sum'),
        wickets_taken=('is_bowler_wicket', 'sum'),
        wides_bowled=('wides', lambda x: (x > 0).sum()),
        noballs_bowled=('noballs', lambda x: (x > 0).sum()),
        dots_bowled=('runs_off_bat', lambda x: (x == 0).sum()) # Simple dot metric
    ).reset_index()

    # Derived Stats
    bowl_group['overs_bowled'] = bowl_group['legal_balls_bowled'] / 6.0
    bowl_group['economy_rate'] = np.where(bowl_group['overs_bowled'] > 0, 
                                          bowl_group['runs_conceded'] / bowl_group['overs_bowled'], 
                                          0)
    bowl_group['bowling_average'] = np.where(bowl_group['wickets_taken'] > 0, 
                                             bowl_group['runs_conceded'] / bowl_group['wickets_taken'], 
                                             0)
    bowl_group['bowling_strike_rate'] = np.where(bowl_group['wickets_taken'] > 0, 
                                                 bowl_group['legal_balls_bowled'] / bowl_group['wickets_taken'], 
                                                 0)

    # ---------------------------------------------------------
    # 4. EXPORT TO CSV
    # ---------------------------------------------------------
    # Sort for readability
    bat_stats = bat_stats.sort_values(by='runs_scored', ascending=False).round(2)
    bowl_group = bowl_group.sort_values(by='wickets_taken', ascending=False).round(2)

    bat_stats.to_csv(BATTING_STATS_PATH, index=False)
    bowl_group.to_csv(BOWLING_STATS_PATH, index=False)
    
    print(f"\nSuccess! Saved '{BATTING_STATS_PATH}' and '{BOWLING_STATS_PATH}'.")
    print(f"Total Batters tracked: {len(bat_stats)}")
    print(f"Total Bowlers tracked: {len(bowl_group)}")


def merge_player_stats():
    # Load the two CSVs you just generated
    bat_stats = pd.read_csv(BATTING_STATS_PATH)
    bowl_stats = pd.read_csv(BOWLING_STATS_PATH)
    
    # Rename the specific columns so they match for the merge
    bat_stats = bat_stats.rename(columns={'striker': 'player_name'})
    bowl_stats = bowl_stats.rename(columns={'bowler': 'player_name'})
    
    # Perform a Full Outer Join
    # This combines rows where player_name matches, and keeps unmatched rows
    master_stats = pd.merge(bat_stats, bowl_stats, on='player_name', how='outer')
    
    # Replace any NaN values with 0 
    # (e.g., a pure bowler will have NaN for runs_scored, make it 0)
    master_stats = master_stats.fillna(0)
    
    # Save the master file for your C++ engine
    master_stats.to_csv(MASTER_STATS_PATH, index=False)
    print(f"Success! Created '{MASTER_STATS_PATH}'")
    print(f"Total unique players: {len(master_stats)}")
    
if __name__ == "__main__":
    # generate_comprehensive_stats()
    merge_player_stats()