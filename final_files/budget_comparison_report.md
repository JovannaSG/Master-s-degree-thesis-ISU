# Budget Factor Comparison Report

## Setup

- **Algorithms**: DQN (d3rlpy DoubleDQN), CQL (d3rlpy DiscreteCQL), BCQ (d3rlpy DiscreteBCQ), PPO (SB3 PPO via DatasetReplayEnv)
- **Budget factors**: 0.3, 0.5, 0.7, 1.0 (applied as multiplier to `max_bid`)
- **Training**: 50,000 steps per algorithm per factor
- **Evaluation**: Per-action mean return — `per_action_means[a] = mean(reward of all samples with action=a)`; policy score = `per_action_means[policy_pred].mean()`
- **Train/test split**: 80/20 by episode (group ID), `random_state=42`
- **BCQ params**: `action_flexibility=0.3, beta=0.5, lr=6.25e-5` (default)
- **CQL params**: `alpha=0.2, lr=1e-4, gamma=0.95`

## Results Summary

| Factor | Baseline | DQN | CQL | BCQ | PPO |
|--------|----------|-----|-----|-----|-----|
| 0.3 | -0.2165 | **-0.0928** (+0.124) | -0.1183 (+0.098) | -0.0713 (+0.145) | -0.2691 (-0.053) |
| 0.5 | -0.4595 | -0.3512 (+0.108) | -0.4473 (+0.012) | **-0.3489** (+0.111) | -0.4151 (+0.044) |
| 0.7 | -0.5868 | -0.5630 (+0.024) | -0.5563 (+0.031) | **-0.5106** (+0.076) | -0.4757 (+0.111) |
| 1.0 | -0.6869 | -0.6256 (+0.061) | -0.6433 (+0.044) | **-0.6176** (+0.069) | -0.7346 (-0.048) |

## Per-Action Means by Budget Factor

### Factor 0.3
| Action | Mean Return |
|--------|-------------|
| 0 (Decrease bid) | -0.3285 |
| 1 (Slight decrease) | -0.2564 |
| 2 (Keep same) | -0.2322 |
| 3 (Slight increase) | -0.2328 |
| 4 (Increase bid) | **0.0015** |

### Factor 0.5
| Action | Mean Return |
|--------|-------------|
| 0 (Decrease bid) | -0.5401 |
| 1 (Slight decrease) | -0.4977 |
| 2 (Keep same) | -0.4850 |
| 3 (Slight increase) | -0.4910 |
| 4 (Increase bid) | **-0.2678** |

### Factor 0.7
| Action | Mean Return |
|--------|-------------|
| 0 (Decrease bid) | -0.6546 |
| 1 (Slight decrease) | -0.6225 |
| 2 (Keep same) | -0.6126 |
| 3 (Slight increase) | -0.6185 |
| 4 (Increase bid) | **-0.4136** |

### Factor 1.0
| Action | Mean Return |
|--------|-------------|
| 0 (Decrease bid) | -0.7478 |
| 1 (Slight decrease) | -0.7188 |
| 2 (Keep same) | -0.7046 |
| 3 (Slight increase) | -0.7166 |
| 4 (Increase bid) | **-0.5322** |

## Test Action Distributions

### Factor 0.3
| Model | Act 0 | Act 1 | Act 2 | Act 3 | Act 4 | Main action |
|-------|-------|-------|-------|-------|-------|-------------|
| Dataset | 30.2% | 19.9% | 9.3% | 19.4% | 21.3% | 0 |
| DQN | 3.6% | 5.4% | 11.6% | 17.7% | 61.7% | 4 |
| CQL | 4.3% | 13.6% | 2.2% | 27.9% | 52.0% | 4 |
| BCQ | 5.3% | 6.9% | 6.3% | 9.7% | 71.7% | 4 |
| PPO | 36.3% | 7.8% | 50.8% | 5.1% | 0.0% | 2 |

### Factor 0.5
| Model | Act 0 | Act 1 | Act 2 | Act 3 | Act 4 | Main action |
|-------|-------|-------|-------|-------|-------|-------------|
| Dataset | 29.7% | 20.0% | 9.5% | 19.9% | 21.0% | 0 |
| DQN | 8.1% | 3.1% | 13.4% | 11.3% | 64.1% | 4 |
| CQL | 23.6% | 8.0% | 4.1% | 39.4% | 24.9% | 3 |
| BCQ | 2.3% | 11.0% | 12.9% | 9.7% | 64.1% | 4 |
| PPO | 4.8% | 27.9% | 1.7% | 29.8% | 35.9% | 4 |

### Factor 0.7
| Model | Act 0 | Act 1 | Act 2 | Act 3 | Act 4 | Main action |
|-------|-------|-------|-------|-------|-------|-------------|
| Dataset | 29.6% | 20.0% | 9.7% | 19.9% | 20.8% | 0 |
| DQN | 4.1% | 8.3% | 29.6% | 30.9% | 27.1% | 3 |
| CQL | 32.9% | 4.1% | 1.7% | 25.2% | 36.2% | 4 |
| BCQ | 3.8% | 10.3% | 19.3% | 13.6% | 53.0% | 4 |
| PPO | 16.1% | 4.4% | 7.1% | 0.0% | 72.4% | 4 |

### Factor 1.0
| Model | Act 0 | Act 1 | Act 2 | Act 3 | Act 4 | Main action |
|-------|-------|-------|-------|-------|-------|-------------|
| Dataset | 29.7% | 20.1% | 9.6% | 19.9% | 20.7% | 0 |
| DQN | 8.8% | 6.7% | 10.6% | 23.6% | 50.3% | 4 |
| CQL | 16.9% | 25.7% | 1.1% | 13.4% | 42.9% | 4 |
| BCQ | 7.1% | 9.2% | 19.1% | 10.9% | 53.7% | 4 |
| PPO | 76.4% | 0.0% | 14.3% | 7.1% | 2.2% | 0 |

## Analysis

1. **Action 4 (Increase bid)** is consistently the best action across all budget factors — the dataset is biased toward conservative bidding.

2. **BCQ** is the most consistent performer across all four budget factors, showing positive improvement over the dataset baseline in every case.

3. **DQN** excels at tight budgets (0.3), achieving the second-best absolute score, but degrades at higher budgets.

4. **CQL** with `alpha=0.2` performs near-DQN at budget 0.3 but degrades significantly at 0.5 (only +0.012 improvement). Higher `alpha` (more conservative) hurts performance.

5. **PPO is unreliable**: it beats BCQ at factor 0.7 (+0.111) but collapses at 1.0 (-0.048) and 0.3 (-0.053), and never picks the best action (4) as its primary action at 0.3.

6. All successful policies converge toward **Action 4 (Increase bid)** as the dominant choice, since it has the highest per-action mean return in the dataset.

## Files

- `results/budget/budget_comparison.json` — full per-factor results
- `results/budget/budget_comparison.png` — comparison plot
- `results/budget/factor_0.3/`, `factor_0.5/`, `factor_0.7/`, `factor_1.0/` — per-factor detailed results
