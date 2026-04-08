import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# This file currently runs a single-innings T20 simulation where team_1 bats first,
# team_2 bowls by the configured 20-over sequence, and each ball outcome is sampled
# from an HMM (New/Settled/Aggressive states) using player stats from clustered_player_stats.csv.

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
    def __init__(self, batting_lineup, bowling_sequence, stats_repository, seed=None):
        self.batting_lineup = batting_lineup
        self.bowling_sequence = bowling_sequence
        self.stats_repository = stats_repository

        self.rng = np.random.default_rng(seed)
        self.hmm = BatterHMM(self.rng)

        self.score = 0
        self.wickets = 0
        self.balls_bowled = 0

        self.striker_idx = 0
        self.non_striker_idx = 1
        self.next_batter_idx = 2

        self.batter_states = {name: "New" for name in batting_lineup}
        self.batter_balls_faced = {name: 0 for name in batting_lineup}

    @classmethod
    def from_config(cls, config_path, stats_path, batting_team_key="team_1", bowling_team_key="team_2", seed=None):
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
        )

    def current_over(self):
        return self.balls_bowled // 6

    def is_innings_over(self):
        return self.wickets >= 10 or self.balls_bowled >= 120

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
        outcome = self.hmm.sample_outcome(next_state, striker_stats, bowler_stats)

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
        }

    def display_scoreboard(self):
        overs = self.balls_bowled // 6
        balls = self.balls_bowled % 6
        striker = self.batting_lineup[self.striker_idx] if not self.is_innings_over() else "-"
        bowler = self.bowling_sequence[self.current_over()] if not self.is_innings_over() else "-"

        print(f"Score: {self.score}/{self.wickets} | Overs: {overs}.{balls}")
        print(f"Striker: {striker} vs Bowler: {bowler}\n")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "match_config.json"
    stats_path = repo_root / "data" / "clustered_player_stats.csv"

    match = Match.from_config(
        config_path=config_path,
        stats_path=stats_path,
        batting_team_key="team_1",
        bowling_team_key="team_2",
        seed=None,
    )

    summary = match.simulate_innings(verbose=False)
    print(
        f"Innings Complete -> {summary['score']}/{summary['wickets']} "
        f"in {summary['balls'] // 6}.{summary['balls'] % 6} overs"
    )