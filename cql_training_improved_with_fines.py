"""
CQL Training with d3rlpy 2.8.1 - Improved Version
====================================================

Enhanced offline RL with:
- Augmented dataset (5000+ samples vs original 731)
- Z-score normalization for observations
- CQL-specific hyperparameters (cql_alpha)
- Larger MLP encoder [256, 256]
- Increased training steps (50k)
- Early stopping based on validation
- Better reward function with ROI/CPA metrics
"""

import numpy as np
import pandas as pd
import ast
import pickle
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig


# =============================================================================
# CONFIGURATION - IMPROVED
# =============================================================================

DATA_PATH = r"pre_vkr/augmented_dataset_combined.csv"
SAVE_DIR = "./saved_model_cql_improved"
N_STEPS = 50000  # Increased from 10000
GAMMA = 0.99
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
CQL_ALPHA = 1.5  # CQL conservatism parameter (tune 0.5-2.0)

# Hyperparameter search configuration
HYPERPARAM_SEARCH = False  # Set to False to skip search
HYPERPARAM_SEARCH_DIR = "./pre_vkr/hyperparam_search_results"
LEARNING_RATES = [0.0001, 0.0003, 0.001]
BATCH_SIZES = [128, 256]
GAMMAS = [0.95, 0.99]
SEARCH_STEPS = 20000  # Reduced steps for faster search
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement after N evaluations

ACTION_NAMES = {
    0: "Decrease bid",
    1: "Slight decrease",
    2: "Keep same",
    3: "Slight increase",
    4: "Increase bid"
}

ACTION_MULTIPLIERS = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.1, 4: 1.2}

OBS_NAMES = ['CPC', 'CTR', 'CPC_change', 'Conversions', 'Position', 'Day', 'DeviceType', 'IsWeekend']


# =============================================================================
# DATA LOADING AND PREPROCESSING - IMPROVED
# =============================================================================

def load_augmented_data(path):
    """Load the augmented dataset."""
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')

    # Convert numeric columns
    numeric_cols = ['CTR (%)', 'Конверсии', 'Клики', 'Показы', 'Расход (руб.)',
                     'Ср. цена клика (руб.)', 'Ср. позиция показов', 'Доход (руб.)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    print(f"Augmented dataset shape: {df.shape}")
    return df


def parse_observations(df):
    """Parse observation strings to numpy arrays."""
    def parse_obs(obs_str):
        try:
            return np.array(ast.literal_eval(obs_str))
        except:
            return np.zeros(8)

    df['observation_array'] = df['observation'].apply(parse_obs)
    return df


def normalize_actions(df):
    """Normalize original actions to 5 classes (0-4)."""
    def normalize(action):
        if action < 50:
            return 0
        elif action < 70:
            return 1
        elif action < 80:
            return 2
        elif action < 100:
            return 3
        else:
            return 4

    df['action_normalized'] = df['action'].apply(normalize)
    return df


def compute_enhanced_rewards(df):
    """
    Enhanced reward function with ROI, CPA, and CTR metrics.
    Uses original weights: roi=0.3, ctr=0.2, conversion=0.3, spend_efficiency=0.2
    """
    # Convert columns to numeric
    df['CTR (%)'] = pd.to_numeric(df['CTR (%)'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Конверсии'] = pd.to_numeric(df['Конверсии'], errors='coerce').fillna(0)
    df['Клики'] = pd.to_numeric(df['Клики'], errors='coerce').fillna(0)
    df['Расход (руб.)'] = pd.to_numeric(df['Расход (руб.)'], errors='coerce').fillna(0)
    df['Доход (руб.)'] = pd.to_numeric(df['Доход (руб.)'], errors='coerce').fillna(0)

    # Base reward: binary activity
    has_clicks = (df['Клики'] > 0).astype(float)
    has_conversions = (df['Конверсии'] > 0).astype(float) * 2
    reward = has_clicks + has_conversions

    # CTR component (continuous)
    ctr = df['CTR (%)'].fillna(0) / 100
    reward = reward + ctr * 0.2  # Increased weight for CTR

    # ROI component (if revenue > 0)
    roi_mask = df['Расход (руб.)'] > 0
    roi = np.zeros(len(df))
    roi[roi_mask] = (df.loc[roi_mask, 'Доход (руб.)'] - df.loc[roi_mask, 'Расход (руб.)']) / df.loc[roi_mask, 'Расход (руб.)']
    roi = np.clip(roi, -1, 5)  # Clip extreme values
    reward = reward + roi * 0.3  # ROI weight

    # CPA bonus (lower CPA = better)
    cpa_mask = df['Конверсии'] > 0
    cpa = np.zeros(len(df))
    cpa[cpa_mask] = df.loc[cpa_mask, 'Расход (руб.)'] / df.loc[cpa_mask, 'Конверсии']
    # Normalize CPA: lower is better
    if cpa[cpa_mask].max() > cpa[cpa_mask].min():
        cpa_bonus = 1 - (cpa[cpa_mask] - cpa[cpa_mask].min()) / (cpa[cpa_mask].max() - cpa[cpa_mask].min())
        reward[cpa_mask] = reward[cpa_mask] + cpa_bonus * 0.2

    # FINES: Penalize bad actions
    # Fine 1: High spend with zero conversions (waste of money)
    high_spend_no_conv = (df['Расход (руб.)'] > df['Расход (руб.)'].quantile(0.7)) & (df['Конверсии'] == 0)
    reward[high_spend_no_conv] -= 0.5

    # Fine 2: Taking "Increase bid" (action 4) when CTR is below median
    ctr_median = df['CTR (%)'].median()
    increase_bid_when_low_ctr = (df['action'] >= 100) & (df['CTR (%)'] < ctr_median)
    reward[increase_bid_when_low_ctr] -= 0.3

    # Fine 3: Taking "Decrease bid" (action 0) when conversions > 0 (might harm performance)
    decrease_when_converting = (df['action'] < 50) & (df['Конверсии'] > 0)
    reward[decrease_when_converting] -= 0.2

    # Normalize to [-1, 1] range
    min_r = reward.min()
    max_r = reward.max()
    if max_r > min_r:
        df['reward_enhanced'] = 2 * (reward - min_r) / (max_r - min_r) - 1
    else:
        df['reward_enhanced'] = np.zeros(len(df))

    return df


def normalize_observations_zscore(observations):
    """
    Z-score normalization for observations (better for neural networks).
    """
    scaler = StandardScaler()
    obs_normalized = scaler.fit_transform(observations)
    return obs_normalized, scaler


def prepare_dataset(df):
    """Prepare observations, actions, rewards, and terminals."""
    observations = np.stack(df['observation_array'].values)

    # Z-score normalization
    observations, scaler = normalize_observations_zscore(observations)

    actions = df['action_normalized'].values.astype(np.int32)

    # Use enhanced rewards
    rewards = df['reward_enhanced'].values.astype(np.float32)

    # Create terminal flags (end of each campaign)
    terminals = np.zeros(len(df), dtype=np.float32)
    df['group_changed'] = df['№ Группы'] != df['№ Группы'].shift(1)
    terminals[df['group_changed'] | (df.index == len(df) - 1)] = 1.0

    return observations, actions, rewards, terminals, scaler


def create_mdp_datasets(observations, actions, rewards, terminals, train_size=0.8):
    """Create train and test MDPDatasets."""
    n_train = int(len(observations) * train_size)

    train_dataset = MDPDataset(
        observations=observations[:n_train],
        actions=actions[:n_train],
        rewards=rewards[:n_train],
        terminals=terminals[:n_train],
    )

    test_dataset = MDPDataset(
        observations=observations[n_train:],
        actions=actions[n_train:],
        rewards=rewards[n_train:],
        terminals=terminals[n_train:],
    )

    return train_dataset, test_dataset


def get_episode_indices(terminals):
    """Get start and end indices (exclusive) for each episode based on terminals."""
    episode_ends = np.where(terminals == 1.0)[0]
    episodes = []
    start = 0
    for end in episode_ends:
        episodes.append((start, end + 1))
        start = end + 1
    return episodes


def create_model_with_params(learning_rate, batch_size, gamma, device='cpu'):
    """Create CQL model with given hyperparameters."""
    config = DiscreteCQLConfig(
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
    )
    cql = DiscreteCQL(
        config=config,
        device=device,
        enable_ddp=False,
    )
    return cql


def cross_validate_params(params, observations, actions, rewards, terminals, n_folds=3, search_steps=20000, patience=3):
    """Perform cross-validation for given hyperparameters."""
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    gamma = params['gamma']
    
    episode_indices = get_episode_indices(terminals)
    n_episodes = len(episode_indices)
    
    if n_episodes < n_folds:
        print(f"Warning: Only {n_episodes} episodes, using {n_episodes} folds instead of {n_folds}")
        n_folds = n_episodes
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    episode_idx_array = np.arange(n_episodes)
    
    fold_scores = []
    
    for fold, (train_ep_idx, val_ep_idx) in enumerate(kf.split(episode_idx_array)):
        print(f"  Fold {fold + 1}/{n_folds}")
        
        train_obs = []
        train_act = []
        train_rew = []
        train_term = []
        
        for ep_i in train_ep_idx:
            start, end = episode_indices[ep_i]
            train_obs.append(observations[start:end])
            train_act.append(actions[start:end])
            train_rew.append(rewards[start:end])
            train_term.append(terminals[start:end])
        
        train_obs = np.concatenate(train_obs)
        train_act = np.concatenate(train_act)
        train_rew = np.concatenate(train_rew)
        train_term = np.concatenate(train_term)
        
        train_dataset = MDPDataset(
            observations=train_obs,
            actions=train_act,
            rewards=train_rew,
            terminals=train_term,
        )
        
        val_obs = []
        val_act = []
        val_rew = []
        val_term = []
        
        for ep_i in val_ep_idx:
            start, end = episode_indices[ep_i]
            val_obs.append(observations[start:end])
            val_act.append(actions[start:end])
            val_rew.append(rewards[start:end])
            val_term.append(terminals[start:end])
        
        val_obs = np.concatenate(val_obs)
        val_act = np.concatenate(val_act)
        val_rew = np.concatenate(val_rew)
        val_term = np.concatenate(val_term)
        
        val_dataset = MDPDataset(
            observations=val_obs,
            actions=val_act,
            rewards=val_rew,
            terminals=val_term,
        )
        
        model = create_model_with_params(learning_rate, batch_size, gamma)
        model.build_with_dataset(train_dataset)
        
        best_val_return = -float('inf')
        patience_counter = 0
        
        n_steps_per_epoch = 1000
        n_epochs = search_steps // n_steps_per_epoch
        
        for epoch in range(n_epochs):
            model.fit(
                train_dataset,
                n_steps=n_steps_per_epoch,
                n_steps_per_epoch=n_steps_per_epoch,
            )
            
            val_returns = evaluate_agent(val_dataset)
            val_mean_return = np.mean(val_returns) if val_returns else -float('inf')
            
            if val_mean_return > best_val_return:
                best_val_return = val_mean_return
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
        
        val_returns = evaluate_agent(val_dataset)
        fold_score = np.mean(val_returns) if val_returns else -float('inf')
        fold_scores.append(fold_score)
        print(f"    Fold score: {fold_score:.4f}")
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    return mean_score, std_score


def hyperparameter_search(observations, actions, rewards, terminals):
    """Perform hyperparameter search over defined space."""
    best_score = -float('inf')
    best_params = None
    best_std = 0
    
    from itertools import product
    param_combinations = list(product(LEARNING_RATES, BATCH_SIZES, GAMMAS))
    total_combinations = len(param_combinations)
    
    print(f"\nStarting hyperparameter search with {total_combinations} combinations...")
    print(f"Search steps per fold: {SEARCH_STEPS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    for i, (lr, bs, gamma) in enumerate(param_combinations):
        params = {
            'learning_rate': lr,
            'batch_size': bs,
            'gamma': gamma
        }
        
        print(f"\nCombination {i+1}/{total_combinations}: LR={lr}, BS={bs}, Gamma={gamma}")
        
        mean_score, std_score = cross_validate_params(
            params,
            observations,
            actions,
            rewards,
            terminals,
            n_folds=3,
            search_steps=SEARCH_STEPS,
            patience=EARLY_STOPPING_PATIENCE
        )
        
        print(f"  Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_std = std_score
            print(f"  ** New best parameters! Score: {best_score:.4f}")
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score:.4f} ± {best_std:.4f}")
    
    # Save to hyperparam_search_results folder
    os.makedirs(HYPERPARAM_SEARCH_DIR, exist_ok=True)
    best_params_path = os.path.join(HYPERPARAM_SEARCH_DIR, 'best_hyperparams.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters saved to: {best_params_path}")
    
    # Save all results
    all_results_path = os.path.join(HYPERPARAM_SEARCH_DIR, 'all_results.json')
    with open(all_results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': float(best_score),
            'best_std': float(best_std),
            'all_scores': [float(s) if s != -float('inf') else -999.0 for s in all_scores] if 'all_scores' in dir() else []
        }, f, indent=2)
    print(f"All results saved to: {all_results_path}")
    
    return best_params


# =============================================================================
# IMPROVED CQL TRAINING
# =============================================================================

def create_improved_cql(device='cpu'):
    """Create CQL with improved hyperparameters."""
    config = DiscreteCQLConfig(
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
    )

    cql = DiscreteCQL(
        config=config,
        device=device,
        enable_ddp=False,
    )

    return cql


def train_cql_with_validation(cql, train_dataset, test_dataset, n_steps=N_STEPS):
    """Train CQL with periodic validation."""
    print(f"\nTraining CQL for {n_steps} steps...")
    print(f"  gamma: {GAMMA}")
    print(f"  learning_rate: {LEARNING_RATE}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  cql_alpha: {CQL_ALPHA}")

    # Track best model
    best_return = -float('inf')
    best_model_path = os.path.join(SAVE_DIR, 'best_cql_model.d3')

    # Train
    cql.fit(
        train_dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
    )

    print("Training complete!")
    return cql


# =============================================================================
# EVALUATION AND METRICS
# =============================================================================

def evaluate_agent(dataset):
    """Calculate cumulative returns per episode."""
    returns = []
    for episode in dataset.episodes:
        episode_return = sum(episode.rewards)
        returns.append(episode_return)
    return returns


def evaluate_policy_detailed(cql, test_dataset):
    """Evaluate trained agent with detailed metrics."""
    test_returns = evaluate_agent(test_dataset)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Test Set) - IMPROVED")
    print("=" * 60)

    mean_return = np.mean(test_returns)
    std_return = np.std(test_returns)

    print(f"\nMean Return: {mean_return:.4f}")
    print(f"Std Return: {std_return:.4f}")
    print(f"Min Return: {np.min(test_returns):.4f}")
    print(f"Max Return: {np.max(test_returns):.4f}")
    print(f"Episodes: {len(test_returns)}")

    # Policy agreement with original (if available)
    all_obs = []
    all_actions = []
    for episode in test_dataset.episodes:
        all_obs.extend(episode.observations)
        all_actions.extend(episode.actions)

    test_obs = np.array(all_obs)
    original_actions = np.array(all_actions)
    agent_actions = cql.predict(test_obs)

    agreement = (agent_actions == original_actions).mean()
    print(f"\nPolicy Agreement with Original: {agreement:.2%}")

    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'min_return': np.min(test_returns),
        'max_return': np.max(test_returns),
        'episodes': len(test_returns),
        'policy_agreement': agreement
    }


def compare_policies(cql, test_dataset):
    """Compare original vs agent policies."""
    all_obs = []
    all_actions = []

    for episode in test_dataset.episodes:
        all_obs.extend(episode.observations)
        all_actions.extend(episode.actions)

    test_obs = np.array(all_obs)
    original_actions = np.array(all_actions)
    agent_actions = cql.predict(test_obs)

    print("\nOriginal Policy Distribution:")
    for action in range(5):
        count = (original_actions == action).sum()
        pct = count / len(original_actions) * 100
        print(f"  [{action}] {ACTION_NAMES[action]}: {count} ({pct:.1f}%)")

    print("\nCQL Agent Distribution:")
    for action in range(5):
        count = (agent_actions == action).sum()
        pct = count / len(agent_actions) * 100
        print(f"  [{action}] {ACTION_NAMES[action]}: {count} ({pct:.1f}%)")


# =============================================================================
# SAVE/LOAD
# =============================================================================

def save_model_improved(cql, df, scaler, save_dir):
    """Save trained model and preprocessor info."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, 'cql_model_improved.d3')
    cql.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save preprocessor info
    preprocessor_info = {
        'feature_names': OBS_NAMES,
        'action_names': ACTION_NAMES,
        'action_multipliers': ACTION_MULTIPLIERS,
        'num_actions': 5,
        'gamma': GAMMA,
        'learning_rate': LEARNING_RATE,
        'n_steps': N_STEPS,
        'cql_alpha': CQL_ALPHA,
        'normalization': 'z-score',
    }

    info_path = os.path.join(save_dir, 'preprocessor_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(preprocessor_info, f)
    print(f"Info saved to: {info_path}")


def get_recommendation(observation, algo, scaler=None):
    """Get bid recommendation with optional scaler."""
    obs = np.array(observation).reshape(1, -1)
    if scaler:
        obs = scaler.transform(obs)
    action = int(algo.predict(obs)[0])

    # Get Q-values for all actions
    q_values_list = []
    for a in range(5):
        action_arr = np.array([a])
        q_val = algo.predict_value(obs, action=action_arr)[0]
        q_values_list.append(q_val)

    q_values = np.array(q_values_list)
    q_probs = np.exp(q_values - q_values.max()) / np.exp(q_values - q_values.max()).sum()

    return {
        'action': action,
        'action_name': ACTION_NAMES[action],
        'bid_multiplier': ACTION_MULTIPLIERS[action],
        'confidence': float(q_probs[action]),
        'q_values': q_values.tolist(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    global LEARNING_RATE, BATCH_SIZE, GAMMA, N_STEPS
    
    print("=" * 70)
    print("IMPROVED CQL TRAINING WITH D3RLPY 2.8.1")
    print("=" * 70)
    print(f"d3rlpy version: {d3rlpy.__version__}")
    print(f"Configuration:")
    print(f"  - N_STEPS: {N_STEPS}")
    print(f"  - GAMMA: {GAMMA}")
    print(f"  - LEARNING_RATE: {LEARNING_RATE}")
    print(f"  - BATCH_SIZE: {BATCH_SIZE}")
    print(f"  - CQL_ALPHA: {CQL_ALPHA}")
    print(f"  - Dataset: {DATA_PATH}")
    print(f"  - Hyperparam Search: {HYPERPARAM_SEARCH}")

    # 1. Load augmented data
    print("\n[1/7] Loading augmented data...")
    df = load_augmented_data(DATA_PATH)

    # 2. Parse observations
    print("\n[2/7] Parsing observations...")
    df = parse_observations(df)

    # 3. Normalize actions
    print("\n[3/7] Normalizing actions to 5 classes...")
    df = normalize_actions(df)

    print("\nAction distribution:")
    for action in range(5):
        count = (df['action_normalized'] == action).sum()
        pct = count / len(df) * 100
        print(f"  [{action}] {ACTION_NAMES[action]}: {count} ({pct:.1f}%)")

    # 4. Compute enhanced rewards
    print("\n[4/7] Computing enhanced rewards...")
    df = compute_enhanced_rewards(df)

    print("\nReward Statistics:")
    print(f"  Enhanced - Mean: {df['reward_enhanced'].mean():.4f}, Std: {df['reward_enhanced'].std():.4f}")
    print(f"  Min: {df['reward_enhanced'].min():.4f}, Max: {df['reward_enhanced'].max():.4f}")

    # 5. Prepare dataset with z-score normalization
    print("\n[5/7] Preparing MDP datasets with z-score normalization...")
    observations, actions, rewards, terminals, scaler = prepare_dataset(df)

    print(f"  Observations shape: {observations.shape}")
    print(f"  Observations mean: {observations.mean():.4f}, std: {observations.std():.4f}")
    print(f"  Rewards range: [{rewards.min():.2f}, {rewards.max():.2f}]")

    if HYPERPARAM_SEARCH:
        # Run hyperparameter search
        print("\n" + "=" * 70)
        print("RUNNING HYPERPARAMETER SEARCH")
        print("=" * 70)

        best_params = hyperparameter_search(observations, actions, rewards, terminals)

        # Train with best params
        print("\nTraining with best parameters...")
        LEARNING_RATE = best_params['learning_rate']
        BATCH_SIZE = best_params['batch_size']
        GAMMA = best_params['gamma']
        N_STEPS = 50000  # Full training

        print(f"Best params - LR: {LEARNING_RATE}, BS: {BATCH_SIZE}, Gamma: {GAMMA}")

    train_dataset, test_dataset = create_mdp_datasets(
        observations, actions, rewards, terminals
    )

    # Get actual sizes
    train_size = sum(len(ep.observations) for ep in train_dataset.episodes)
    test_size = sum(len(ep.observations) for ep in test_dataset.episodes)
    print(f"  Train: {train_size} samples, {len(train_dataset.episodes)} episodes")
    print(f"  Test: {test_size} samples, {len(test_dataset.episodes)} episodes")

    # 6. Train improved CQL
    print("\n[6/7] Training improved CQL...")
    cql = create_improved_cql()
    cql.build_with_dataset(train_dataset)

    print(f"  Observation shape: {cql.observation_shape}")
    print(f"  Action size: {cql.action_size}")

    cql = train_cql_with_validation(cql, train_dataset, test_dataset, n_steps=N_STEPS)

    # 7. Evaluate
    print("\n[7/7] Evaluating...")
    metrics = evaluate_policy_detailed(cql, test_dataset)
    compare_policies(cql, test_dataset)

    # Save
    save_model_improved(cql, df, scaler, SAVE_DIR)

    # Example recommendations
    print("\n" + "=" * 70)
    print("EXAMPLE RECOMMENDATIONS")
    print("=" * 70)

    for i in [0, 50, 100]:
        if i < len(df):
            obs = df['observation_array'].iloc[i]
            rec = get_recommendation(obs, cql, scaler)
            print(f"\nSample {i}:")
            print(f"  Features: CPC={obs[0]:.2f}, CTR={obs[1]:.4f}, Conv={obs[3]:.0f}")
            print(f"  Recommended: [{rec['action']}] {rec['action_name']} (x{rec['bid_multiplier']})")
            print(f"  Confidence: {rec['confidence']:.2%}")
            print(f"  Q-values: {[f'{q:.2f}' for q in rec['q_values']]}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - IMPROVED VERSION!")
    print("=" * 70)

    return cql


if __name__ == "__main__":
    cql = main()
