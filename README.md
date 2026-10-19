# Offline RL for Ad-Campaign Bid Optimization (Master's Thesis, ISU)

Offline reinforcement learning research for optimizing bids in Yandex.Direct advertising campaigns.
The task is framed as a **contextual bandit / MDP** where the agent adjusts a CPC bid (5 discrete
actions) and is rewarded based on ROI, clicks, conversions, and CTR.

## Repository Structure

```
.
├── Data pipeline
│   ├── clean_and_filter.py        # filter zero-conversion rows, drop unused columns
│   └── augment_dataset.py         # Gaussian-noise augmentation of observations (~731 → 3500+ rows)
│
├── Core experiments
│   ├── compare_algorithms.py      # CQL vs Double DQN vs Ensemble CQL vs SAC (d3rlpy),
│   │                              # ROI-based reward, state-dependent RandomForest evaluation
│   └── best_hyperparams.json      # best CQL params (lr=1e-4, batch=128, gamma=0.95)
│
├── final_files/                   # main experiment: budget-constrained comparison
│   ├── compare_ppo_dqn.py         # core module: data loading, rewards, budget mechanism,
│   │                              # DatasetReplayEnv (Gymnasium), DQN/BCQ/CQL/PPO training
│   ├── compare_budget_full.py     # loops over budget factors 0.3/0.5/0.7/1.0
│   ├── search_bcq_params.py       # BCQ grid search (40 combos)
│   ├── search_cql_params.py       # CQL grid search (20 combos)
│   ├── budget_comparison_experiment.md / budget_comparison_report.md
│   ├── bcq_algorithm_description.md
│   ├── draw_bcq_*.py, *.puml, *.dot   # diagram sources (PlantUML / Graphviz / matplotlib)
│   └── results/                   # JSON + plots for all runs
│
├── hparam_search/                 # CQL alpha × lr search report
├── alpha_search_results/          # CQL alpha sweep (0.1 … 10.0) + trained .d3 models
├── validation_results/            # full CQL validation run report + best model
├── early_stopping_results/        # early-stopping run (best epoch checkpoint)
└── comparison_results_budget/     # budget comparison artifacts (JSON, PNG, HTML, scaler)
```

CSV datasets are git-ignored (`.gitignore`: `*.csv`, `__pycache__/`); scripts expect them under
paths like `pre_vkr/augmented_dataset_merged.csv`.

## Problem Setup

- **Observations (9 features)**: CPC, CTR, CPC_change, Conversions, Position, Day, DeviceType,
  IsWeekend, BudgetRemaining
- **Actions (5 discrete)**: bid multipliers — Decrease (×0.8), Slight decrease (×0.9),
  Keep same (×1.0), Slight increase (×1.1), Increase (×1.2)
- **Reward**: composite of clicks + conversions, CTR bonus (0.2), ROI bonus (0.3), CPA bonus (0.2),
  minus fines (spend without conversion 0.5, raise bid on low CTR 0.3, cut bid with good
  conversions 0.2), normalized to `[-1, 1]`
- **Budget factor**: each episode's budget is `budget_factor × historical spend`;
  `budget_remaining` is appended as the 9th feature and transitions with no budget left are
  dropped — simulating tighter ad budgets
- **Split**: 80/20 train/test by episode (campaign group ID), `random_state=42`
- **Evaluation**: per-action mean return — `score(π) = mean(per_action_means[π(s_i)])` over test
  episodes; compared against the dataset (behavioral) baseline

## Algorithms

| Algorithm | Library | Key parameters |
|-----------|---------|----------------|
| Double DQN | d3rlpy | γ=0.99, lr=3e-4, batch=256 |
| Discrete CQL | d3rlpy | γ=0.95, lr=1e-4, batch=128, α=0.2 |
| Discrete BCQ | d3rlpy | γ=0.99, lr=6.25e-5, batch=256, action_flexibility=0.3, β=0.5 |
| PPO | Stable-Baselines3 | 100k timesteps via `DatasetReplayEnv` |

All offline algorithms train for 50,000 gradient steps per budget factor.
`compare_algorithms.py` additionally implements a 3-member **Ensemble CQL**, per-epoch logging
with early stopping on best Δ reward, and a state-dependent **RandomForest ROI predictor** for
counterfactual policy evaluation.

## Requirements

```
python >= 3.10
d3rlpy
stable-baselines3
gymnasium
torch
scikit-learn
pandas, numpy
matplotlib
```

## Usage

```bash
# 1. Clean the merged dataset (keeps 15% of zero-conversion rows)
python clean_and_filter.py

# 2. Augment observations with Gaussian noise
python augment_dataset.py

# 3. Baseline algorithm comparison (CQL / DQN / Ensemble CQL / SAC)
python compare_algorithms.py

# 4. Budget-constrained comparison (DQN / CQL / BCQ / PPO)
python final_files/compare_budget_full.py

# Hyperparameter searches
python final_files/search_bcq_params.py     # 40 BCQ combos
python final_files/search_cql_params.py     # 20 CQL combos

# Diagrams
plantuml final_files/budget_experiment_flow.puml
dot -Tpng -Gdpi=300 final_files/bcq_architecture.dot -o final_files/bcq_architecture.png
python final_files/draw_bcq_uml.py
```

Update the `DATA_PATH` / `SAVE_DIR` constants at the top of each script to point at your data
and output locations.

## Results (Budget Comparison)

Score = per-action mean return on test episodes; parentheses = improvement over dataset baseline.

| Factor | Baseline | DQN | CQL | BCQ | PPO |
|--------|----------|-----|-----|-----|-----|
| 0.3 | -0.2165 | **-0.0928** (+0.124) | -0.1183 (+0.098) | -0.0713 (+0.145) | -0.2691 (-0.053) |
| 0.5 | -0.4595 | -0.3512 (+0.108) | -0.4473 (+0.012) | **-0.3489** (+0.111) | -0.4151 (+0.044) |
| 0.7 | -0.5868 | -0.5630 (+0.024) | -0.5563 (+0.031) | **-0.5106** (+0.076) | -0.4757 (+0.111) |
| 1.0 | -0.6869 | -0.6256 (+0.061) | -0.6433 (+0.044) | **-0.6176** (+0.069) | -0.7346 (-0.048) |

Key findings:

1. **BCQ is the only algorithm with positive improvement at every budget factor.**
2. Offline RL helps most under tight budgets (factor 0.3); gains shrink as budget grows.
3. **PPO is unstable** with fixed-trajectory replay (negative at 0.3 and 1.0).
4. Learned policies collapse toward **Action 4 (Increase bid)** — it has the highest per-action
   mean return; the behavioral dataset is overly conservative.

Detailed numbers, action distributions, and analysis:
`final_files/budget_comparison_experiment.md`, `final_files/budget_comparison_report.md`.

Earlier experiments (CQL alpha sweeps, hyperparameter grid, early stopping vs. human baseline)
are recorded in `hparam_search/`, `alpha_search_results/`, `validation_results/`, and
`early_stopping_results/`.

## References

- Fujimoto, Meger, Precup (2019). *Off-Policy Deep Reinforcement Learning without Exploration.* ICML.
- Kumar et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS.
- d3rlpy: https://github.com/takuseno/d3rlpy
- Stable-Baselines3: https://stable-baselines3.readthedocs.io
