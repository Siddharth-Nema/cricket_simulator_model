import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np  
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "raw_data"

def download_cricsheet_data(format_type="t20s"):
    """
    Downloads and extracts ball-by-ball CSV data from cricsheet.org.
    format_type can be 't20s', 'odis', 'tests', 'ipl', etc.
    """
    print(f"Downloading {format_type} data from Cricsheet...")
    
    # Cricsheet bulk download URL for CSVs
    url = f"https://cricsheet.org/downloads/{format_type}_csv2.zip"
    
    # Create a directory for the data
    data_dir = RAW_DATA_DIR / format_type
    os.makedirs(data_dir, exist_ok=True)
    
    # Fetch the zip file
    response = requests.get(url)
    if response.status_code == 200:
        # Unzip the contents in memory and extract
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(str(data_dir))
        print(f"Success! Data extracted to {data_dir}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def inspect_data(format_type="t20s"):
    """
    Reads all downloaded CSVs and concatenates them into a single pandas DataFrame
    for easy analysis and EM clustering later.
    """
    data_dir = RAW_DATA_DIR / format_type
    all_files = [str(data_dir / f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print(f"Found {len(all_files)} match files. Loading the first 50 for inspection...")
    
    # We load just a subset initially so it doesn't eat all your RAM
    df_list = [pd.read_csv(file) for file in all_files[:50]]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("\nData Columns Available:")
    print(combined_df.columns.tolist())
    
    print("\nSample Data (First 5 rows):")
    print(combined_df[['match_id', 'striker', 'bowler', 'runs_off_bat', 'extras']].head())


def calculate_features(format_type="t20s"):
    data_dir = RAW_DATA_DIR / format_type
    
    # THE FIX: Filter out the metadata _info.csv files
    all_files = [str(data_dir / f) for f in os.listdir(data_dir) 
                 if f.endswith('.csv') and not f.endswith('_info.csv')]
    
    print(f"Found {len(all_files)} ball-by-ball files. Loading first 50 matches for feature extraction...")
    
    # Load a subset of files to build the prototype quickly
    df_list = [pd.read_csv(file) for file in all_files[:50]]
    df = pd.concat(df_list, ignore_index=True)
    
    # 1. Define the Match Phase based on the 'ball' column
    conditions = [
        (df['ball'] < 6.0),
        (df['ball'] >= 6.0) & (df['ball'] < 15.0),
        (df['ball'] >= 15.0)
    ]
    choices = ['Powerplay', 'Middle', 'Death']
    df['phase'] = np.select(conditions, choices, default='Unknown')
    
    # 2. Add helper columns for calculation
    df['is_dot'] = (df['runs_off_bat'] == 0)
    df['is_boundary'] = df['runs_off_bat'].isin([4, 6])
    
    # Handle wicket_type safely (it is NaN if no wicket fell on that delivery)
    if 'wicket_type' in df.columns:
        df['is_wicket'] = df['wicket_type'].notna()
    else:
        df['is_wicket'] = False
        
    # 3. Calculate Batsman Features
    # Group by Striker and Phase
    bat_stats = df.groupby(['striker', 'phase']).agg(
        balls_faced=('ball', 'count'),
        total_runs=('runs_off_bat', 'sum'),
        dots=('is_dot', 'sum'),
        boundaries=('is_boundary', 'sum'),
        dismissals=('is_wicket', 'sum')
    ).reset_index()
    
    # Filter out small sample sizes (e.g., fewer than 20 balls faced) to prevent statistical noise
    bat_stats = bat_stats[bat_stats['balls_faced'] >= 20].copy()
    
    # Calculate final Feature Vectors [x_1, x_2, x_3, x_4]
    bat_stats['strike_rate'] = (bat_stats['total_runs'] / bat_stats['balls_faced']) * 100
    bat_stats['dot_pct'] = bat_stats['dots'] / bat_stats['balls_faced']
    bat_stats['boundary_pct'] = bat_stats['boundaries'] / bat_stats['balls_faced']
    bat_stats['dismissal_rate'] = bat_stats['dismissals'] / bat_stats['balls_faced']
    
    print("\nBatsman Feature Vectors (Sample):")
    # Displaying the calculated features rounded to 3 decimal places
    print(bat_stats[['striker', 'phase', 'strike_rate', 'dot_pct', 'boundary_pct', 'dismissal_rate']].round(3).head(10))


if __name__ == "__main__":
    # download_cricsheet_data("t20s")
    # inspect_data("t20s")
    calculate_features("t20s")