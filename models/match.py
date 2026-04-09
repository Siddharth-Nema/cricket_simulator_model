import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# This file runs a T20 simulation engine where each ball is sampled from an HMM
# (New/Settled/Aggressive states) using player stats from clustered_player_stats.csv.
# It supports both innings and winner logic for a full match simulation.


@dataclass
class DeliveryResult:
    outcome: str
    runs: int
    is_wicket: bool
    striker: str
    bowler: str
    state: str


class PlayerStatsRepository:
    """Fast in-memory lookup for player stats and archetype labels."""

    def __init__(self, clustered_stats_path):
        df = pd.read_csv(clustered_stats_path)
        self.df = df.set_index("player_name", drop=False)
        self.default_row = self._build_default_row(df)

    @staticmethod
    def _build_default_row(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        defaults = df[numeric_cols].median(numeric_only=True).to_dict()
        defaults["player_name"] = "UNKNOWN"
        defaults.setdefault("batting_cluster", -1)
        defaults.setdefault("bowling_cluster", -1)
        return defaults

    def get(self, player_name):
        if player_name in self.df.index:
            return self.df.loc[player_name].to_dict()
        row = dict(self.default_row)
        row["player_name"] = player_name
        return row


class BatterHMM:
    """
    Hidden states: New, Settled, Aggressive.
    Emissions: 0, 1, 2, 3, 4, 6, W.
    """

    STATES = ["New", "Settled", "Aggressive"]
    OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def transition(self, current_state, balls_faced):
        idx = self.STATES.index(current_state)

        if balls_faced < 8:
            matrix = np.array(
                [
                    [0.70, 0.25, 0.05],
                    [0.20, 0.70, 0.10],
                    [0.15, 0.45, 0.40],
                ]
            )
        elif balls_faced < 20:
            matrix = np.array(
                [
                    [0.45, 0.45, 0.10],
                    [0.10, 0.75, 0.15],
                    [0.10, 0.45, 0.45],
                ]
            )
        else:
            matrix = np.array(
                [
                    [0.35, 0.40, 0.25],
                    [0.05, 0.65, 0.30],
                    [0.05, 0.35, 0.60],
                ]
            )

        return self.rng.choice(self.STATES, p=matrix[idx])

    def emission_probs(self, state, batter_stats, bowler_stats):
        bat_balls = max(float(batter_stats.get("balls_faced", 0.0)), 1.0)
        bowl_balls = max(float(bowler_stats.get("legal_balls_bowled", 0.0)), 1.0)

        dot_rate_bat = np.clip(float(batter_stats.get("dots", 0.0)) / bat_balls, 0.15, 0.80)
        boundary_bat = np.clip(float(batter_stats.get("boundary_pct", 0.0)), 0.02, 0.40)
        dismiss_bat = np.clip(float(batter_stats.get("times_dismissed", 0.0)) / bat_balls, 0.005, 0.12)

        economy = np.clip(float(bowler_stats.get("economy_rate", 7.5)), 4.5, 11.0)
        wicket_rate_bowl = np.clip(float(bowler_stats.get("wickets_taken", 0.0)) / bowl_balls, 0.005, 0.10)

        econ_factor = (economy - 7.5) / 7.5
        p_w = np.clip(0.5 * dismiss_bat + 0.5 * wicket_rate_bowl, 0.01, 0.20)
        p_boundary = np.clip(boundary_bat * (1.0 + 0.35 * econ_factor), 0.03, 0.50)
        p_dot = np.clip(dot_rate_bat * (1.0 - 0.25 * econ_factor), 0.10, 0.85)

        if state == "New":
            p_dot *= 1.20
            p_w *= 1.20
            p_boundary *= 0.75
        elif state == "Aggressive":
            p_dot *= 0.75
            p_w *= 1.30
            p_boundary *= 1.30

        p_dot = float(np.clip(p_dot, 0.08, 0.90))
        p_w = float(np.clip(p_w, 0.01, 0.30))
        p_boundary = float(np.clip(p_boundary, 0.02, 0.60))

        remaining = max(1.0 - (p_dot + p_w + p_boundary), 0.01)
        p1 = remaining * 0.64
        p2 = remaining * 0.26
        p3 = remaining * 0.10

        p4 = p_boundary * 0.78
        p6 = p_boundary * 0.22

        probs = np.array([p_dot, p1, p2, p3, p4, p6, p_w], dtype=float)
        probs = probs / probs.sum()
        return probs

    def sample_outcome(self, state, batter_stats, bowler_stats):
        probs = self.emission_probs(state, batter_stats, bowler_stats)
        return self.rng.choice(self.OUTCOMES, p=probs)


class Match:
    def __init__(
        self,
        batting_lineup,
        bowling_sequence,
        stats_repository,
        seed=None,
        target_score=None,
        importance_lambda=0.0,
    ):
        self.batting_lineup = batting_lineup
        self.bowling_sequence = bowling_sequence
        self.stats_repository = stats_repository
        self.target_score = target_score
        self.importance_lambda = float(importance_lambda)

        self.rng = np.random.default_rng(seed)
        self.hmm = BatterHMM(self.rng)

        self.score = 0
        self.wickets = 0
        self.balls_bowled = 0

        self.striker_idx = 0
        self.non_striker_idx = 1
        self.next_batter_idx = 2
        self.log_importance_weight = 0.0

        self.batter_states = {name: "New" for name in batting_lineup}
        self.batter_balls_faced = {name: 0 for name in batting_lineup}

    @classmethod
    def from_config(
        cls,
        config_path,
        stats_path,
        batting_team_key="team_1",
        bowling_team_key="team_2",
        seed=None,
        target_score=None,
        importance_lambda=0.0,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        batting_lineup = config[batting_team_key]["lineup"]
        bowling_sequence = config[bowling_team_key]["bowling_sequence"]

        if len(bowling_sequence) != 20:
            raise ValueError("bowling_sequence must contain exactly 20 overs for T20.")

        repo = PlayerStatsRepository(stats_path)
        return cls(
            batting_lineup=batting_lineup,
            bowling_sequence=bowling_sequence,
            stats_repository=repo,
            seed=seed,
            target_score=target_score,
            importance_lambda=importance_lambda,
        )

    def current_over(self):
        return self.balls_bowled // 6

    def is_innings_over(self):
        chase_complete = self.target_score is not None and self.score >= self.target_score
        return self.wickets >= 10 or self.balls_bowled >= 120 or chase_complete

    def _current_players(self):
        striker = self.batting_lineup[self.striker_idx]
        bowler = self.bowling_sequence[self.current_over()]
        return striker, bowler

    def _bring_next_batter(self):
        if self.next_batter_idx >= len(self.batting_lineup):
            return
        self.striker_idx = self.next_batter_idx
        self.next_batter_idx += 1

    def _swap_strike(self):
        self.striker_idx, self.non_striker_idx = self.non_striker_idx, self.striker_idx

    def simulate_ball(self):
        if self.is_innings_over():
            return None

        striker, bowler = self._current_players()
        striker_stats = self.stats_repository.get(striker)
        bowler_stats = self.stats_repository.get(bowler)

        current_state = self.batter_states[striker]
        balls_seen = self.batter_balls_faced[striker]

        next_state = self.hmm.transition(current_state, balls_seen)
        base_probs = self.hmm.emission_probs(next_state, striker_stats, bowler_stats)

        if self.importance_lambda == 0.0:
            outcome = self.rng.choice(self.hmm.OUTCOMES, p=base_probs)
        else:
            proposal_probs = self._build_importance_proposal(base_probs, self.importance_lambda)
            sampled_idx = int(self.rng.choice(len(self.hmm.OUTCOMES), p=proposal_probs))
            outcome = self.hmm.OUTCOMES[sampled_idx]
            p = float(base_probs[sampled_idx])
            q = float(proposal_probs[sampled_idx])
            self.log_importance_weight += np.log(max(p, 1e-15)) - np.log(max(q, 1e-15))

        self.batter_states[striker] = next_state
        self.balls_bowled += 1
        self.batter_balls_faced[striker] += 1

        is_wicket = outcome == "W"
        runs = 0 if is_wicket else int(outcome)

        if is_wicket:
            self.wickets += 1
            if not self.is_innings_over():
                self._bring_next_batter()
        else:
            self.score += runs
            if runs % 2 == 1:
                self._swap_strike()

        if self.balls_bowled % 6 == 0 and not self.is_innings_over():
            self._swap_strike()

        return DeliveryResult(
            outcome=outcome,
            runs=runs,
            is_wicket=is_wicket,
            striker=striker,
            bowler=bowler,
            state=next_state,
        )

    def _build_importance_proposal(self, base_probs, importance_lambda):
        # Exponential tilting toward higher-run outcomes to reduce rare-tail variance.
        outcome_runs = np.array([0, 1, 2, 3, 4, 6, 0], dtype=float)
        tilted = base_probs * np.exp(importance_lambda * outcome_runs)
        total = float(np.sum(tilted))
        if total <= 0.0:
            return base_probs
        return tilted / total

    def simulate_innings(self, verbose=False):
        deliveries = []
        while not self.is_innings_over():
            result = self.simulate_ball()
            if result is None:
                break
            deliveries.append(result)
            if verbose:
                over = (self.balls_bowled - 1) // 6
                ball = (self.balls_bowled - 1) % 6 + 1
                print(
                    f"{over}.{ball} | {result.bowler} to {result.striker} | "
                    f"State={result.state} | Outcome={result.outcome} | "
                    f"Score={self.score}/{self.wickets}"
                )
        return {
            "score": self.score,
            "wickets": self.wickets,
            "balls": self.balls_bowled,
            "deliveries": deliveries,
            "log_weight": self.log_importance_weight,
            "weight": float(np.exp(np.clip(self.log_importance_weight, -700.0, 700.0))),
        }

    def display_scoreboard(self):
        overs = self.balls_bowled // 6
        balls = self.balls_bowled % 6
        striker = self.batting_lineup[self.striker_idx] if not self.is_innings_over() else "-"
        bowler = self.bowling_sequence[self.current_over()] if not self.is_innings_over() else "-"

        print(f"Score: {self.score}/{self.wickets} | Overs: {overs}.{balls}")
        print(f"Striker: {striker} vs Bowler: {bowler}\\n")


def simulate_match(config_path, stats_path, seed=None, verbose=False, importance_lambda=0.0):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    rng = np.random.default_rng(seed)

    team_keys = ["team_1", "team_2"]
    toss_winner_key = rng.choice(team_keys)
    toss_loser_key = "team_2" if toss_winner_key == "team_1" else "team_1"
    toss_decision = rng.choice(["bat", "bowl"])

    if toss_decision == "bat":
        first_batting_key = toss_winner_key
        second_batting_key = toss_loser_key
    else:
        first_batting_key = toss_loser_key
        second_batting_key = toss_winner_key

    stats_repo = PlayerStatsRepository(stats_path)

    innings_1 = Match(
        batting_lineup=config[first_batting_key]["lineup"],
        bowling_sequence=config[second_batting_key]["bowling_sequence"],
        stats_repository=stats_repo,
        seed=None if seed is None else seed + 1,
        importance_lambda=importance_lambda,
    )
    summary_1 = innings_1.simulate_innings(verbose=verbose)
    target = summary_1["score"] + 1

    innings_2 = Match(
        batting_lineup=config[second_batting_key]["lineup"],
        bowling_sequence=config[first_batting_key]["bowling_sequence"],
        stats_repository=stats_repo,
        seed=None if seed is None else seed + 2,
        target_score=target,
        importance_lambda=importance_lambda,
    )
    summary_2 = innings_2.simulate_innings(verbose=verbose)

    first_batting_name = config[first_batting_key]["name"]
    second_batting_name = config[second_batting_key]["name"]

    if summary_2["score"] > summary_1["score"]:
        winner = second_batting_name
        margin = f"won by {10 - summary_2['wickets']} wickets"
    elif summary_2["score"] < summary_1["score"]:
        winner = first_batting_name
        margin = f"won by {summary_1['score'] - summary_2['score']} runs"
    else:
        winner = "Tie"
        margin = "scores level"

    return {
        "team_1": config["team_1"]["name"],
        "team_2": config["team_2"]["name"],
        "first_batting": first_batting_name,
        "second_batting": second_batting_name,
        "toss_winner": config[toss_winner_key]["name"],
        "toss_decision": toss_decision,
        "innings_1": summary_1,
        "innings_2": summary_2,
        "winner": winner,
        "margin": margin,
        "target": target,
        "log_weight": summary_1["log_weight"] + summary_2["log_weight"],
        "weight": summary_1["weight"] * summary_2["weight"],
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "match_config.json"
    stats_path = repo_root / "data" / "clustered_player_stats.csv"

    result = simulate_match(config_path=config_path, stats_path=stats_path, seed=None, verbose=False)

    print(f"Toss -> {result['toss_winner']} won the toss and chose to {result['toss_decision']}")

    print(
        f"Innings 1 -> {result['first_batting']}: {result['innings_1']['score']}/{result['innings_1']['wickets']} "
        f"in {result['innings_1']['balls'] // 6}.{result['innings_1']['balls'] % 6} overs"
    )
    print(
        f"Innings 2 -> {result['second_batting']}: {result['innings_2']['score']}/{result['innings_2']['wickets']} "
        f"in {result['innings_2']['balls'] // 6}.{result['innings_2']['balls'] % 6} overs"
    )
    print(f"Result -> {result['winner']} ({result['margin']})")
