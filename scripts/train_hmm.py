import pandas as pd
import numpy as np
import os
from pathlib import Path
from hmmlearn import hmm

def get_batter_sequences(data_dir=None):
    """Extracts chronological ball-by-ball sequences for every batter's innings."""
    if data_dir is None:
        # Dynamically point to the root of your project
        repo_root = Path(__file__).resolve().parents[1]
        
        # Pointing to raw_data based on your traceback
        data_dir = repo_root / "raw_data" / "t20s"  
        
    print(f"Loading raw ball-by-ball data from: {data_dir} ...")
    
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.csv') and not f.endswith('_info.csv')]
    
    # Training on the first 500 matches for speed
    df_list = [pd.read_csv(f) for f in all_files[:500]]
    df = pd.concat(df_list, ignore_index=True)
    
    # Map outcomes to discrete integers for hmmlearn
    # 0=0, 1=1, 2=2, 3=3, 4=4, 5=6, 6=W
    def map_outcome(row):
        if pd.notna(row.get('wicket_type')) and row['wicket_type'] not in ['run out', 'retired hurt', 'obstructing the field']:
            return 6 # Wicket
        elif row['runs_off_bat'] == 6:
            return 5 # Six
        elif row['runs_off_bat'] in [0, 1, 2, 3, 4]:
            return row['runs_off_bat']
        return 0 # Default to dot for extras handling
        
    df['encoded_outcome'] = df.apply(map_outcome, axis=1)
    
    # Group by match and batter to get sequences
    sequences = []
    sequence_lengths = []
    
    grouped = df.groupby(['match_id', 'striker'])
    for _, group in grouped:
        # FIX: Changed 'overs' to 'ball' to match Cricsheet format
        seq = group.sort_values(['innings', 'ball'])['encoded_outcome'].values
        if len(seq) >= 5: # Only train on innings where they faced at least 5 balls
            sequences.append(seq.reshape(-1, 1))
            sequence_lengths.append(len(seq))
            
    # hmmlearn requires a single concatenated array and a list of lengths
    X = np.concatenate(sequences)
    return X, sequence_lengths

def train_baum_welch():
    X, lengths = get_batter_sequences()
    print(f"Extracted {len(lengths)} valid innings sequences. Total balls: {len(X)}")
    
    print("Running Expectation-Maximization (Baum-Welch)...")
    # n_components = 3 (New, Settled, Aggressive)
    model = hmm.CategoricalHMM(n_components=3, n_iter=100, random_state=42, init_params="te")
    
    # Initialize starting probabilities: 100% chance they start in State 0
    model.startprob_ = np.array([1.0, 0.0, 0.0])
    
    # Fit the model
    model.fit(X, lengths)
    
    print(f"\nModel converged: {model.monitor_.converged}")
    print("\n--- LEARNED TRANSITION MATRIX ---")
    print(np.round(model.transmat_, 3))
    
    print("\n--- LEARNED EMISSION MATRIX ---")
    print("Columns: [0, 1, 2, 3, 4, 6, W]")
    print(np.round(model.emissionprob_, 3))

if __name__ == "__main__":
    train_baum_welch()