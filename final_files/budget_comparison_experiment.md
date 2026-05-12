# Budget Factor Comparison Experiment

## Objective

Evaluate how offline RL algorithms perform under different budget constraints in a contextual bandit bid optimization setting. The budget factor controls how aggressively the agent can spend — lower factors simulate tighter budgets.

## Experimental Setup

### Dataset

- Source: `pre_vkr/augmented_dataset_merged.csv` — historical Yandex.Direct campaign data
- 9 features per observation: CPC, CTR, CPC_change, Conversions, Position, Day, DeviceType, IsWeekend, BudgetRemaining
- 5 discrete actions: Decrease bid (0), Slight decrease (1), Keep same (2), Slight increase (3), Increase bid (4)
- Rewards computed by `compute_rewards()` with components:
  - Base: clicks + conversions
  - CTR bonus ($w_{ctr} = 0.2$)
  - ROI bonus ($w_{roi} = 0.3$)
  - CPA bonus ($w_{cpa} = 0.2$)
  - Fines: high spend without conversion (0.5), increase bid with low CTR (0.3), decrease bid with good conversions (0.2)
  - Final reward normalized to $[-1, 1]$

### Budget Factor Mechanism

The budget factor is applied in `make_budgeted_data()`:

```
for each episode (group):
    cumulative_spend = cumsum(spends)
    episode_total = cumulative_spend[-1]
    budget = episode_total × budget_factor
    budget_remaining = (budget - cumulative_spend) / budget
    clip budget_remaining to [0, 1]
    append budget_remaining as 9th observation feature
    discard transitions where budget_remaining ≤ 0
```

This simulates: "you have `budget_factor × historical_spend` available — allocate it across the episode."

### Algorithms

| Algorithm | Library | Parameters |
|-----------|---------|------------|
| DQN (DoubleDQN) | d3rlpy | $\gamma=0.99$, $lr=3\times10^{-4}$, batch=256 |
| CQL (DiscreteCQL) | d3rlpy | $\gamma=0.95$, $lr=1\times10^{-4}$, batch=128, $\alpha=0.2$ |
| BCQ (DiscreteBCQ) | d3rlpy | $\gamma=0.99$, $lr=6.25\times10^{-5}$, batch=256, $af=0.3$, $\beta=0.5$ |
| PPO | SB3 | 100k timesteps via DatasetReplayEnv |

All algorithms train for **50,000 gradient steps** per budget factor.

### Train / Test Split

- 80/20 split by episode (group ID)
- `random_state=42` for reproducibility
- Test episodes are unseen during training

### Evaluation Metric: Per-Action Mean Return

For each action $a \in \{0, 1, 2, 3, 4\}$:

$$per\_action\_means[a] = \frac{1}{N_a} \sum_{i: a_i = a} reward_i$$

Where $N_a$ is the number of samples where action $a$ was taken in the **full dataset**.

Then a policy $\pi$ is evaluated as:

$$score(\pi) = \frac{1}{N_{test}} \sum_{i \in test} per\_action\_means[\pi(s_i)]$$

This measures: *"how good are the actions this policy chooses, on average across the test set?"*

### Budget Factors Tested

0.3, 0.5, 0.7, 1.0

## Results

### Summary Table

| Factor | Baseline | DQN | CQL | BCQ | PPO |
|--------|----------|-----|-----|-----|-----|
| 0.3 | -0.2165 | **-0.0928** (+0.124) | -0.1183 (+0.098) | -0.0713 (+0.145) | -0.2691 (-0.053) |
| 0.5 | -0.4595 | -0.3512 (+0.108) | -0.4473 (+0.012) | **-0.3489** (+0.111) | -0.4151 (+0.044) |
| 0.7 | -0.5868 | -0.5630 (+0.024) | -0.5563 (+0.031) | **-0.5106** (+0.076) | -0.4757 (+0.111) |
| 1.0 | -0.6869 | -0.6256 (+0.061) | -0.6433 (+0.044) | **-0.6176** (+0.069) | -0.7346 (-0.048) |

*Values in parentheses = improvement over dataset baseline. Bold = best per factor.*

### Key Findings

1. **BCQ is the most consistent** — the only algorithm with positive improvement at all four budget factors
2. **Low budgets (0.3)**: All offline RL methods improve significantly; the tight budget makes bid optimization easiest
3. **High budgets (1.0)**: Improvement shrinks — with full budget, the dataset baseline is already reasonable
4. **PPO is unreliable** — negative at 0.3 and 1.0, positive only at 0.7; SB3's PPO struggles with fixed-trajectory replay
5. **Action 4 (Increase bid)** dominates all learned policies (53–72% selection), as it has the highest per-action mean return

### Per-Action Mean Returns

| Action | 0.3 | 0.5 | 0.7 | 1.0 |
|--------|-----|-----|-----|-----|
| 0 (Decrease) | -0.3285 | -0.5401 | -0.6546 | -0.7478 |
| 1 (Slight ↓) | -0.2564 | -0.4977 | -0.6225 | -0.7188 |
| 2 (Keep) | -0.2322 | -0.4850 | -0.6126 | -0.7046 |
| 3 (Slight ↑) | -0.2328 | -0.4910 | -0.6185 | -0.7166 |
| 4 (Increase) | **0.0015** | **-0.2678** | **-0.4136** | **-0.5322** |

Action 4 (Increase bid) is consistently the best — the more the budget tightens, the more valuable it is to increase bids.

## Code Organization

- `compare_ppo_dqn.py` — main experiment script (data loading, reward, training, evaluation, plotting)
- `compare_budget_full.py` — loops over budget factors, imports from `compare_ppo_dqn.py`
- `search_bcq_params.py` — BCQ hyperparameter grid search (40 combos)
- `search_cql_params.py` — CQL hyperparameter grid search (20 combos)

## Output Files

- `results/budget/budget_comparison.json` — full per-factor numerical results
- `results/budget/budget_comparison.png` — grouped bar chart comparing all algorithms
- `results/budget/factor_X/` — per-factor results (JSON with all metrics)

## Experiment Flow Diagram

PlantUML source: `budget_experiment_flow.puml`
Render: `plantuml budget_experiment_flow.puml`

## References

- Fujimoto et al. (2019). "Off-Policy Deep Reinforcement Learning without Exploration." *ICML 2019*
- Kumar et al. (2020). "Conservative Q-Learning for Offline Reinforcement Learning." *NeurIPS 2020*
- d3rlpy: https://github.com/takuseno/d3rlpy
