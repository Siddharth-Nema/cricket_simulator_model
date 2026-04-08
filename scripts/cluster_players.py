import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MASTER_STATS_PATH = DATA_DIR / "master_player_stats.csv"
CLUSTERED_STATS_PATH = DATA_DIR / "clustered_player_stats.csv"

def apply_gmm_clustering(file_path):
    print("Loading player statistics for EM clustering...")
    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 1. BATTING CLUSTERS (GMM & Expectation-Maximization)
    # ---------------------------------------------------------
    # Filter batters with enough data to avoid statistical noise
    batters = df[df['balls_faced'] > 30].copy()
    bat_features = ['strike_rate', 'batting_average', 'boundary_pct']
    
    # Scale features to mean=0, variance=1
    scaler_bat = StandardScaler()
    X_bat = scaler_bat.fit_transform(batters[bat_features])
    
    # Initialize GMM with 4 components (Archetypes)
    gmm_bat = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    batters['batting_cluster'] = gmm_bat.fit_predict(X_bat)
    
    print(f"Batting EM Algorithm converged in {gmm_bat.n_iter_} iterations.")

    # ---------------------------------------------------------
    # 2. BOWLING CLUSTERS (GMM & Expectation-Maximization)
    # ---------------------------------------------------------
    bowlers = df[df['legal_balls_bowled'] > 30].copy()
    bowl_features = ['economy_rate', 'bowling_strike_rate', 'bowling_average']
    
    scaler_bowl = StandardScaler()
    X_bowl = scaler_bowl.fit_transform(bowlers[bowl_features])
    
    gmm_bowl = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    bowlers['bowling_cluster'] = gmm_bowl.fit_predict(X_bowl)

    print(f"Bowling EM Algorithm converged in {gmm_bowl.n_iter_} iterations.")

    # ---------------------------------------------------------
    # 3. MERGE BACK & EXPORT
    # ---------------------------------------------------------
    # Map the clusters back to the main dataframe. 
    # Give a default cluster of -1 to players with insufficient data.
    df['batting_cluster'] = df['player_name'].map(batters.set_index('player_name')['batting_cluster']).fillna(-1)
    df['bowling_cluster'] = df['player_name'].map(bowlers.set_index('player_name')['bowling_cluster']).fillna(-1)
    
    # Convert cluster IDs to integers for cleaner lookups later
    df['batting_cluster'] = df['batting_cluster'].astype(int)
    df['bowling_cluster'] = df['bowling_cluster'].astype(int)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLUSTERED_STATS_PATH, index=False)
    print(f"\nSuccess! Saved '{CLUSTERED_STATS_PATH}'")
    
    # Quick sanity check output
    print("\nSample Batting Archetypes:")
    print(df[df['batting_cluster'] != -1][['player_name', 'batting_cluster', 'strike_rate', 'batting_average']].head(10))

if __name__ == "__main__":
    apply_gmm_clustering(MASTER_STATS_PATH)