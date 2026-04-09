from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.match import simulate_match


def logsumexp(log_values):
    max_log = np.max(log_values)
    return max_log + np.log(np.sum(np.exp(log_values - max_log)))


def effective_sample_size(weights):
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    if sum_w2 == 0:
        return 0.0
    return (sum_w ** 2) / sum_w2


def run_standard_mc(config_path, stats_path, n_runs=2000):
    team_1_wins = 0
    first_innings_scores = []

    for i in range(n_runs):
        result = simulate_match(config_path=config_path, stats_path=stats_path, seed=i, importance_lambda=0.0)
        first_innings_scores.append(result["innings_1"]["score"])
        if result["winner"] == result["team_1"]:
            team_1_wins += 1

    return {
        "team_1_win_prob": team_1_wins / n_runs,
        "first_innings_mean": float(np.mean(first_innings_scores)),
        "first_innings_std": float(np.std(first_innings_scores)),
    }


def run_importance_sampling(config_path, stats_path, n_runs=2000, importance_lambda=0.10):
    indicators = []
    first_innings_scores = []
    log_weights = []

    for i in range(n_runs):
        result = simulate_match(
            config_path=config_path,
            stats_path=stats_path,
            seed=i,
            importance_lambda=importance_lambda,
        )
        indicators.append(1.0 if result["winner"] == result["team_1"] else 0.0)
        first_innings_scores.append(float(result["innings_1"]["score"]))
        log_weights.append(float(result["log_weight"]))

    indicators = np.array(indicators, dtype=float)
    first_innings_scores = np.array(first_innings_scores, dtype=float)
    log_weights = np.array(log_weights, dtype=float)

    log_z = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_z)

    weighted_team_1_win_prob = float(np.sum(normalized_weights * indicators))
    weighted_first_innings_mean = float(np.sum(normalized_weights * first_innings_scores))

    raw_weights = np.exp(np.clip(log_weights, -700.0, 700.0))
    ess = float(effective_sample_size(raw_weights))

    return {
        "team_1_win_prob": weighted_team_1_win_prob,
        "first_innings_mean": weighted_first_innings_mean,
        "ess": ess,
        "ess_ratio": ess / len(log_weights),
    }


if __name__ == "__main__":
    config_path = REPO_ROOT / "match_config.json"
    stats_path = REPO_ROOT / "data" / "clustered_player_stats.csv"

    n_runs = 2000
    importance_lambda = 0.01

    standard = run_standard_mc(config_path, stats_path, n_runs=n_runs)
    is_result = run_importance_sampling(
        config_path,
        stats_path,
        n_runs=n_runs,
        importance_lambda=importance_lambda,
    )

    print("Standard Monte Carlo")
    print(f"  Runs: {n_runs}")
    print(f"  Team 1 Win Probability: {standard['team_1_win_prob']:.4f}")
    print(f"  First Innings Mean: {standard['first_innings_mean']:.2f}")
    print(f"  First Innings Std: {standard['first_innings_std']:.2f}")

    print("\nImportance Sampling Monte Carlo")
    print(f"  Runs: {n_runs}")
    print(f"  Lambda: {importance_lambda:.3f}")
    print(f"  Weighted Team 1 Win Probability: {is_result['team_1_win_prob']:.4f}")
    print(f"  Weighted First Innings Mean: {is_result['first_innings_mean']:.2f}")
    print(f"  Effective Sample Size (ESS): {is_result['ess']:.1f}")
    print(f"  ESS Ratio: {is_result['ess_ratio']:.4f}")
