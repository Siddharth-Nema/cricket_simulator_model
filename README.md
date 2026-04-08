To keep this prototype manageable for your lab tomorrow while still hitting the key concepts from your syllabus, we will use **three primary mathematical/ML models**.

We will split them into two phases: what happens _before_ the match (data prep), and what happens _during_ the match (inside the class we just built).

### 1. The Pre-Match Model: Gaussian Mixture Model (GMM) via EM

- **What it is:** We will use a GMM trained with the Expectation-Maximization (EM) algorithm on the `master_player_stats.csv` we created.
- **Where it fits:** This runs once, offline, before your simulation ever starts.
- **Why we use it:** To solve data sparsity. If Virat Kohli has only faced a specific new bowler 3 times, we can't build a reliable probability distribution from that. The GMM clusters players into hidden "Archetypes" (e.g., Batters: _Anchors, Power-Hitters_; Bowlers: _Economy Spinners, Strike Pacers_). We calculate the ball probabilities based on how _Power-Hitters_ fare against _Strike Pacers_, rather than relying on sparse individual data.

### 2. The In-Game Model: Hidden Markov Model (HMM)

- **What it is:** A Markov chain where the "true" state of the system is hidden, but it emits observable outputs.
- **Where it fits:** Right inside your `Match` class, specifically in the `simulate_ball()` logic.
- **Why we use it:** To model a batter's "Form" or "Eye". A batter doesn't hit with the same probabilities all the time.
  - **Hidden States:** $Z \in \{\text{New to Crease}, \text{Settled}, \text{Aggressive}\}$.
  - **Emissions (Observable):** The runs scored $X \in \{0, 1, 2, 3, 4, 6, W\}$.
  - **Mechanism:** When a batter first comes in, they are in the "New" state (high probability of dot balls and wickets). As they face more balls, the Markov transition matrix gives them a higher probability of moving to the "Settled" state (lower wicket risk, higher strike rotation).

### 3. The Execution Model: Monte Carlo Simulation & Discrete Sampling

- **What it is:** Running a stochastic process repeatedly with random sampling to find expected values.
- **Where it fits:** This is the wrapper around your `Match` class. You instantiate your class and play the 120 balls using **Sampling from Discrete Distributions** (`numpy.random.choice` based on the HMM's current probability weights). You do this 10,000 times.
- **Why we use it:** Cricket is highly volatile. A single simulation run is meaningless. By running a Monte Carlo loop, you can plot the distribution of total scores to find the Expected Total (Mean) and the Variance.

### Summary for the Professor

If asked at 3 PM how your engine works, you can say:
_"We pre-cluster players using a **GMM trained via EM** to generalize matchup probabilities. During the match, we treat the batter's progression as a **Hidden Markov Model**, sampling discrete delivery outcomes based on their current latent state, and run the entire engine inside a **Monte Carlo Simulation** to determine the expected score distribution."_

Does this sound like a solid, defensible setup for your class? If so, we can write the logic for `simulate_ball()` to actually use these probabilities and update the scoreboard.
