from datetime import datetime
import os
import json
import ast
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.algos import DoubleDQN, DoubleDQNConfig
from d3rlpy.algos import DiscreteSAC, DiscreteSACConfig

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

DATA_PATH = r"pre_vkr/augmented_dataset_merged.csv"
SAVE_DIR = r"pre_vkr/comparison_results_v2"
N_STEPS = 50000
GAMMA = 0.99
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
CQL_ALPHA = 1.0
SAC_TEMPERATURE = 1.0
N_STEPS_PER_EPOCH = 1000
N_EPOCHS = N_STEPS // N_STEPS_PER_EPOCH
N_ENSEMBLE = 3

ACTION_NAMES = {0: "Decrease bid", 1: "Slight decrease", 2: "Keep same", 3: "Slight increase", 4: "Increase bid"}
ACTION_MULTIPLIERS = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.1, 4: 1.2}
OBS_NAMES = ['CPC', 'CTR', 'CPC_change', 'Conversions', 'Position', 'Day', 'DeviceType', 'IsWeekend', 'Spend', 'Revenue', 'ROI']


def load_augmented_data(path):
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    numeric_cols = ['CTR (%)', 'Конверсии', 'Клики', 'Показы', 'Расход (руб.)',
                    'Ср. цена клика (руб.)', 'Ср. позиция показов', 'Доход (руб.)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    print(f"Dataset shape: {df.shape}")
    return df


def parse_observations(df):
    def parse_obs(obs_str):
        try:
            return np.array(ast.literal_eval(obs_str))
        except:
            return np.zeros(8)
    df['observation_array'] = df['observation'].apply(parse_obs)
    return df


def normalize_actions(df):
    def normalize(action):
        if action < 50: return 0
        elif action < 70: return 1
        elif action < 80: return 2
        elif action < 100: return 3
        else: return 4
    df['action_normalized'] = df['action'].apply(normalize)
    return df


def compute_enhanced_rewards(df):
    spend = pd.to_numeric(df['Расход (руб.)'], errors='coerce').fillna(0)
    revenue = pd.to_numeric(df['Доход (руб.)'], errors='coerce').fillna(0)
    profit = revenue - spend
    roi = np.where(spend > 0, profit / spend, 0)
    roi = np.clip(roi, -5, 20)
    df['reward_enhanced'] = roi.astype(np.float32)
    return df


def prepare_dataset(df):
    observations = np.stack(df['observation_array'].values)
    spend = pd.to_numeric(df['Расход (руб.)'], errors='coerce').fillna(0).values.reshape(-1, 1)
    revenue = pd.to_numeric(df['Доход (руб.)'], errors='coerce').fillna(0).values.reshape(-1, 1)
    roi = np.where(spend[:, 0] > 0, (revenue[:, 0] - spend[:, 0]) / spend[:, 0], 0).reshape(-1, 1)
    roi = np.clip(roi, -1, 10)
    observations = np.concatenate([observations, spend, revenue, roi], axis=1)

    scaler = StandardScaler()
    observations = scaler.fit_transform(observations)
    actions = df['action_normalized'].values.astype(np.int32)
    rewards = df['reward_enhanced'].values.astype(np.float32)
    terminals = np.zeros(len(df), dtype=np.float32)
    group_changed = df['№ Группы'] != df['№ Группы'].shift(1)
    terminals[group_changed | (df.index == len(df) - 1)] = 1.0
    return observations, actions, rewards, terminals, scaler


def create_mdp_datasets(observations, actions, rewards, terminals, train_size=0.8):
    n_train = int(len(observations) * train_size)
    train_dataset = MDPDataset(observations=observations[:n_train], actions=actions[:n_train],
                                rewards=rewards[:n_train], terminals=terminals[:n_train])
    test_dataset = MDPDataset(observations=observations[n_train:], actions=actions[n_train:],
                               rewards=rewards[n_train:], terminals=terminals[n_train:])
    return train_dataset, test_dataset, n_train


def compute_action_value_table(df):
    action_values = []
    for a in range(5):
        mask = df['action_normalized'] == a
        if mask.sum() > 0:
            action_values.append(df.loc[mask, 'reward_enhanced'].mean())
        else:
            action_values.append(0.0)
    return action_values


def train_roi_predictor(observations, actions, rewards):
    n_obs_features = observations.shape[1]
    action_onehot = np.zeros((len(actions), 5))
    action_onehot[np.arange(len(actions)), actions] = 1
    X = np.concatenate([observations, action_onehot], axis=1)
    y = rewards
    model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   min_samples_leaf=5, random_state=42,
                                   n_jobs=-1)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = float(r2_score(y, y_pred))
    print(f"  ROI predictor R^2 on training data: {r2:.4f}")
    return model


def predict_counterfactual_roi(model, observation, action):
    action_onehot = np.zeros((1, 5))
    action_onehot[0, int(action)] = 1
    X = np.concatenate([observation.reshape(1, -1), action_onehot], axis=1)
    return float(model.predict(X)[0])


class EnsembledCQL:
    def __init__(self, models):
        self.models = models
        print(f"  EnsembleCQL created with {len(models)} models")

    def predict(self, observations):
        all_q = np.mean([m.predict_value(observations) for m in self.models], axis=0)
        return np.argmax(all_q, axis=1).astype(np.int32)

    def predict_value(self, observations, action=None):
        all_q = np.mean([m.predict_value(observations) for m in self.models], axis=0)
        if action is not None:
            return all_q[np.arange(len(observations)), action]
        return all_q

    def save(self, path_prefix):
        for i, m in enumerate(self.models):
            m.save(f"{path_prefix}_member{i}.d3")


def train_ensemble_cql(train_dataset, n_models=N_ENSEMBLE, n_steps=N_STEPS // 2):
    models = []
    for i in range(n_models):
        seed = 42 + i
        d3rlpy.seed(seed)
        config = DiscreteCQLConfig(batch_size=BATCH_SIZE, gamma=GAMMA, learning_rate=LEARNING_RATE,
                                   alpha=CQL_ALPHA)
        model = DiscreteCQL(config=config, enable_ddp=False)
        model.build_with_dataset(train_dataset)
        model.fit(train_dataset, n_steps=n_steps)
        models.append(model)
        print(f"  Ensemble member {i+1}/{n_models} trained (seed={seed})")
    return EnsembledCQL(models)


def compute_policy_metrics(algo, dataset, df_test, action_values, roi_predictor=None):
    all_obs = []
    for episode in dataset.episodes:
        all_obs.extend(episode.observations)
    test_obs = np.array(all_obs)
    agent_actions = algo.predict(test_obs)

    q_values = algo.predict_value(test_obs, action=agent_actions)
    mean_q = float(np.mean(q_values))

    raw_actions = [a for ep in dataset.episodes for a in ep.actions]
    original_actions = np.array(raw_actions).ravel()
    agreement = float((agent_actions == original_actions).mean())

    actual_rewards = np.array([r for ep in dataset.episodes for r in ep.rewards]).ravel()
    mean_original_reward = float(np.mean(actual_rewards))

    if roi_predictor is not None:
        agent_rewards = np.where(
            agent_actions == np.array(raw_actions[:len(agent_actions)]).ravel(),
            actual_rewards[:len(agent_actions)],
            np.array([predict_counterfactual_roi(roi_predictor, test_obs[i], agent_actions[i]) for i in range(len(agent_actions))])
        )
    else:
        agent_rewards = np.where(
            agent_actions == original_actions,
            actual_rewards[:len(agent_actions)],
            np.array([action_values[int(a)] for a in agent_actions])
        )
    mean_agent_reward = float(np.mean(agent_rewards))

    delta = mean_agent_reward - mean_original_reward

    dist = pd.Series(agent_actions).value_counts().reindex(range(5), fill_value=0)
    total = dist.sum()
    dist_pct = {int(k): float(v / total) for k, v in dist.items()}
    action_mode = dist.idxmax()
    collapse_ratio = float(dist.max() / total)

    return {
        'mean_q': mean_q,
        'mean_reward': mean_agent_reward,
        'mean_original_reward': mean_original_reward,
        'delta_reward': delta,
        'policy_agreement': agreement,
        'action_distribution': dist_pct,
        'action_mode': int(action_mode),
        'collapse_ratio': collapse_ratio,
    }


def train_with_logging(algo, algo_name, train_dataset, test_dataset, df_test, action_values, save_best=False, roi_predictor=None):
    history = {
        'epoch': [], 'mean_q': [], 'mean_reward': [],
        'mean_original_reward': [], 'delta_reward': [],
        'policy_agreement': [], 'action_dist': [], 'collapse_ratio': []
    }

    n_epochs = N_STEPS // N_STEPS_PER_EPOCH
    print(f"\nTraining {algo_name} for {n_epochs} epochs ({N_STEPS} steps)...")

    best_checkpoint = None
    best_delta = -1e9
    best_epoch = -1

    for epoch in range(n_epochs):
        algo.fit(train_dataset, n_steps=N_STEPS_PER_EPOCH, n_steps_per_epoch=N_STEPS_PER_EPOCH)

        metrics = compute_policy_metrics(algo, test_dataset, df_test, action_values, roi_predictor=roi_predictor)

        history['epoch'].append(epoch)
        history['mean_q'].append(metrics['mean_q'])
        history['mean_reward'].append(metrics['mean_reward'])
        history['mean_original_reward'].append(metrics['mean_original_reward'])
        history['delta_reward'].append(metrics['delta_reward'])
        history['policy_agreement'].append(metrics['policy_agreement'])
        history['action_dist'].append(metrics['action_distribution'])
        history['collapse_ratio'].append(metrics['collapse_ratio'])

        if save_best and metrics['delta_reward'] > best_delta:
            best_delta = metrics['delta_reward']
            best_epoch = epoch
            best_checkpoint = os.path.join(SAVE_DIR, f'{algo_name.lower().replace(" ", "_")}_best_checkpoint.pkl')
            with open(best_checkpoint, 'wb') as f:
                pickle.dump(algo, f)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{algo_name}] Epoch {epoch+1}/{n_epochs} | "
                  f"Q: {metrics['mean_q']:.4f} | "
                  f"Reward: {metrics['mean_reward']:.4f} | "
                  f"Delta: {metrics['delta_reward']:.4f} | "
                  f"Agree: {metrics['policy_agreement']:.2%}")

    if save_best and best_checkpoint and os.path.exists(best_checkpoint):
        print(f"\n  [{algo_name}] Early stopping: restored model from epoch {best_epoch+1} (best delta={best_delta:.4f})")
        with open(best_checkpoint, 'rb') as f:
            algo = pickle.load(f)
        os.remove(best_checkpoint)

    return algo, history


def plot_learning_curves(cql_history, dqn_history, save_dir, ensemble_history=None):
    n_rows = 2 if ensemble_history is None else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows))

    if ensemble_history is None:
        axes = axes.reshape(-1, 2)

    epochs = cql_history['epoch']

    axes[0, 0].plot(epochs, cql_history['mean_q'], 'b-', label='CQL', linewidth=2)
    axes[0, 0].plot(epochs, dqn_history['mean_q'], 'r-', label='Double DQN', linewidth=2)
    if ensemble_history is not None:
        axes[0, 0].plot(epochs, ensemble_history['mean_q'], 'g-', label='Ensemble CQL', linewidth=2)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Predicted Q-value')
    axes[0, 0].set_title('Predicted Expected Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, cql_history['mean_reward'], 'b-', label='CQL', linewidth=2)
    axes[0, 1].plot(epochs, dqn_history['mean_reward'], 'r-', label='DQN', linewidth=2)
    if ensemble_history is not None:
        axes[0, 1].plot(epochs, ensemble_history['mean_reward'], 'g-', label='Ensemble CQL', linewidth=2)
    axes[0, 1].plot(epochs, cql_history['mean_original_reward'], 'k--', label='Original (baseline)', linewidth=1.5)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean ROI')
    axes[0, 1].set_title('Policy Value (ROI)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, cql_history['delta_reward'], 'b-', label='CQL vs original', linewidth=2)
    axes[1, 0].plot(epochs, dqn_history['delta_reward'], 'r-', label='DQN vs original', linewidth=2)
    if ensemble_history is not None:
        axes[1, 0].plot(epochs, ensemble_history['delta_reward'], 'g-', label='Ensemble CQL vs original', linewidth=2)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Delta ROI')
    axes[1, 0].set_title('Improvement over Human Policy')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    final_cql_dist = cql_history['action_dist'][-1]
    final_dqn_dist = dqn_history['action_dist'][-1]
    x = np.arange(5)
    width = 0.2
    axes[1, 1].bar(x - width, [final_cql_dist[i] for i in range(5)], width,
                   label='CQL', color='b', alpha=0.7)
    axes[1, 1].bar(x, [final_dqn_dist[i] for i in range(5)], width,
                   label='Double DQN', color='r', alpha=0.7)
    if ensemble_history is not None:
        final_ensemble_dist = ensemble_history['action_dist'][-1]
        axes[1, 1].bar(x + width, [final_ensemble_dist[i] for i in range(5)], width,
                       label='Ensemble CQL', color='g', alpha=0.7)
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_title('Final Policy Distribution')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([ACTION_NAMES[i].split()[0] for i in range(5)])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if ensemble_history is not None:
        axes[2, 0].plot(epochs, cql_history['policy_agreement'], 'b-', label='CQL', linewidth=2)
        axes[2, 0].plot(epochs, dqn_history['policy_agreement'], 'r-', label='DQN', linewidth=2)
        axes[2, 0].plot(epochs, ensemble_history['policy_agreement'], 'g-', label='Ensemble CQL', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Agreement')
        axes[2, 0].set_title('Policy Agreement with Original Actions')
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(epochs, cql_history['collapse_ratio'], 'b-', label='CQL collapse', linewidth=2, alpha=0.5)
        axes[2, 1].plot(epochs, dqn_history['collapse_ratio'], 'r-', label='DQN collapse', linewidth=2, alpha=0.5)
        if ensemble_history is not None:
            axes[2, 1].plot(epochs, ensemble_history['collapse_ratio'], 'g-', label='Ensemble CQL collapse', linewidth=2, alpha=0.5)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Collapse Ratio')
        axes[2, 1].set_title('Action Collapse Over Time')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison_learning_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nLearning curves saved to: {path}")


def plot_reward_distribution(df, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['reward_enhanced'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution (ROI-based)')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, 'reward_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Reward distribution saved to: {path}")


def save_comparison_report(cql_history, dqn_history, cql_metrics, dqn_metrics, save_dir, sac_metrics=None, ensemble_metrics=None):
    report = {
        'comparison_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_params': {
            'data_path': DATA_PATH,
            'n_steps': N_STEPS,
            'gamma': GAMMA,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'cql_alpha': CQL_ALPHA,
            'sac_temperature': SAC_TEMPERATURE,
            'n_ensemble': N_ENSEMBLE,
            'evaluation': 'state-dependent ROI predictor (RandomForest)',
        },
        'cql': {
            'final_mean_q': float(np.mean(cql_history['mean_q'][-5:])),
            'final_mean_reward': float(np.mean(cql_history['mean_reward'][-5:])),
            'final_delta_reward': float(np.mean(cql_history['delta_reward'][-5:])),
            'best_q_epoch': int(np.argmax(cql_history['mean_q'])),
            'best_q': float(max(cql_history['mean_q'])),
            'best_reward_epoch': int(np.argmax(cql_history['mean_reward'])),
            'best_reward': float(max(cql_history['mean_reward'])),
            'final_agreement': cql_metrics['policy_agreement'],
            'action_distribution': cql_metrics['action_distribution'],
            'action_mode': cql_metrics['action_mode'],
            'collapse_ratio': cql_metrics['collapse_ratio'],
            'original_policy_reward': cql_metrics['mean_original_reward'],
        },
        'double_dqn': {
            'final_mean_q': float(np.mean(dqn_history['mean_q'][-5:])),
            'final_mean_reward': float(np.mean(dqn_history['mean_reward'][-5:])),
            'final_delta_reward': float(np.mean(dqn_history['delta_reward'][-5:])),
            'best_q_epoch': int(np.argmax(dqn_history['mean_q'])),
            'best_q': float(max(dqn_history['mean_q'])),
            'best_reward_epoch': int(np.argmax(dqn_history['mean_reward'])),
            'best_reward': float(max(dqn_history['mean_reward'])),
            'final_agreement': dqn_metrics['policy_agreement'],
            'action_distribution': dqn_metrics['action_distribution'],
            'action_mode': dqn_metrics['action_mode'],
            'collapse_ratio': dqn_metrics['collapse_ratio'],
            'original_policy_reward': dqn_metrics['mean_original_reward'],
        },
    }

    if sac_metrics is not None:
        report['sac'] = {
            'final_delta_reward': sac_metrics['delta_reward'],
            'final_agreement': sac_metrics['policy_agreement'],
            'action_distribution': sac_metrics['action_distribution'],
            'action_mode': sac_metrics['action_mode'],
            'collapse_ratio': sac_metrics['collapse_ratio'],
        }

    if ensemble_metrics is not None:
        report['ensemble_cql'] = {
            'final_delta_reward': ensemble_metrics['delta_reward'],
            'final_agreement': ensemble_metrics['policy_agreement'],
            'action_distribution': ensemble_metrics['action_distribution'],
            'action_mode': ensemble_metrics['action_mode'],
            'collapse_ratio': ensemble_metrics['collapse_ratio'],
        }

    path = os.path.join(save_dir, 'comparison_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Comparison report saved to: {path}")

    print("\n" + "=" * 100)
    print("FINAL COMPARISON (State-Dependent Evaluation)")
    print("=" * 100)
    headers = ['CQL', 'Ensemble CQL', 'SAC', 'Double DQN']
    print(f"{'Metric':<35} {'CQL':<15} {'Ens CQL':<15} {'SAC':<15} {'Double DQN':<15}")
    print("-" * 100)
    cql_r = report['cql']
    dqn_r = report['double_dqn']
    sac_r = report.get('sac', {})
    ens_r = report.get('ensemble_cql', {})

    def v(d, k, fmt='.4f'):
        val = d.get(k, 'N/A')
        if isinstance(val, str):
            return f'{val:<15}'
        return f'{val:{fmt}}' if fmt else f'{val:<15}'

    print(f"{'Delta ROI (Agent - Original)':<35} {cql_r['final_delta_reward']:<15.4f} {v(ens_r, 'final_delta_reward'):<15} {v(sac_r, 'final_delta_reward'):<15} {dqn_r['final_delta_reward']:<15.4f}")
    print(f"{'Mean Agent ROI':<35} {cql_r['final_mean_reward']:<15.4f} {'N/A':<15} {'N/A':<15} {dqn_r['final_mean_reward']:<15.4f}")
    print(f"{'Original Policy ROI':<35} {cql_r['original_policy_reward']:<15.4f} {'N/A':<15} {'N/A':<15} {dqn_r['original_policy_reward']:<15.4f}")
    print(f"{'Policy Agreement':<35} {cql_r['final_agreement']:<15.2%} {v(ens_r, 'policy_agreement', '.2%'):<15} {v(sac_r, 'policy_agreement', '.2%'):<15} {dqn_r['final_agreement']:<15.2%}")
    print(f"{'Action Collapse Ratio':<35} {cql_r['collapse_ratio']:<15.2%} {v(ens_r, 'collapse_ratio', '.2%'):<15} {v(sac_r, 'collapse_ratio', '.2%'):<15} {dqn_r['collapse_ratio']:<15.2%}")
    print(f"{'Dominant Action':<35} {ACTION_NAMES[cql_r['action_mode']]:<15} {ACTION_NAMES[ens_r.get('action_mode', -1)] if ens_r.get('action_mode') is not None else 'N/A':<15} {ACTION_NAMES[sac_r.get('action_mode', -1)] if sac_r.get('action_mode') is not None else 'N/A':<15} {ACTION_NAMES[dqn_r['action_mode']]:<15}")
    print("-" * 100)

    return report


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("CQL vs Ensemble CQL vs SAC vs DOUBLE DQN (ROI-based, State-Dependent Eval)")
    print("=" * 70)
    print(f"d3rlpy version: {d3rlpy.__version__}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Steps: {N_STEPS}, Gamma: {GAMMA}, LR: {LEARNING_RATE}, BS: {BATCH_SIZE}")
    print(f"Device: {DEVICE}, Ensemble: {N_ENSEMBLE} CQL models")

    print("\n[1/8] Loading and preprocessing data...")
    df = load_augmented_data(DATA_PATH)
    df = parse_observations(df)
    df = normalize_actions(df)
    df = compute_enhanced_rewards(df)

    print("\nAction distribution in dataset:")
    for a in range(5):
        count = (df['action_normalized'] == a).sum()
        print(f"  [{a}] {ACTION_NAMES[a]}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\nReward stats - Mean: {df['reward_enhanced'].mean():.4f}, Std: {df['reward_enhanced'].std():.4f}")

    plot_reward_distribution(df, SAVE_DIR)

    print("\n[2/8] Preparing MDP datasets (80/20 split)...")
    observations, actions, rewards, terminals, scaler = prepare_dataset(df)
    train_dataset, test_dataset, n_train = create_mdp_datasets(observations, actions, rewards, terminals)

    df_test = df.iloc[n_train:].reset_index(drop=True)

    train_samples = sum(len(ep.observations) for ep in train_dataset.episodes)
    test_samples = sum(len(ep.observations) for ep in test_dataset.episodes)
    print(f"  Train: {train_samples} samples, {len(train_dataset.episodes)} episodes")
    print(f"  Test: {test_samples} samples, {len(test_dataset.episodes)} episodes")
    print(f"  Obs shape: {observations.shape}")

    print("\n[3/8] Training CQL...")
    cql_config = DiscreteCQLConfig(batch_size=BATCH_SIZE, gamma=GAMMA, learning_rate=LEARNING_RATE,
                                    alpha=CQL_ALPHA)
    cql = DiscreteCQL(config=cql_config, device=DEVICE, enable_ddp=False)
    cql.build_with_dataset(train_dataset)
    df_train = df.iloc[:n_train]
    action_values = compute_action_value_table(df_train)
    print(f"\nAction value table (expected ROI per action):")
    for a in range(5):
        print(f"  [{a}] {ACTION_NAMES[a]}: {action_values[a]:.4f}")

    print(f"\nTraining state-dependent ROI predictor on {n_train} training samples...")
    train_obs = observations[:n_train]
    train_acts = actions[:n_train]
    train_rews = rewards[:n_train]
    roi_predictor = train_roi_predictor(train_obs, train_acts, train_rews)

    cql, cql_history = train_with_logging(cql, "CQL", train_dataset, test_dataset, df_test, action_values, save_best=True, roi_predictor=roi_predictor)

    print("\n[4/8] Training Double DQN...")
    dqn_config = DoubleDQNConfig(batch_size=BATCH_SIZE, gamma=GAMMA, learning_rate=LEARNING_RATE)
    dqn = DoubleDQN(config=dqn_config, enable_ddp=False)
    dqn.build_with_dataset(train_dataset)
    dqn, dqn_history = train_with_logging(dqn, "Double DQN", train_dataset, test_dataset, df_test, action_values, roi_predictor=roi_predictor)

    print(f"\n[5/8] Training Ensemble CQL ({N_ENSEMBLE} models, {N_STEPS // 2} steps each)...")
    d3rlpy.seed(42)
    ensemble_cql = train_ensemble_cql(train_dataset, n_models=N_ENSEMBLE, n_steps=N_STEPS // 2)

    print("\n[6/8] Evaluating all policies (state-dependent evaluation)...")
    cql_metrics = compute_policy_metrics(cql, test_dataset, df_test, action_values, roi_predictor=roi_predictor)
    dqn_metrics = compute_policy_metrics(dqn, test_dataset, df_test, action_values, roi_predictor=roi_predictor)
    ensemble_metrics = compute_policy_metrics(ensemble_cql, test_dataset, df_test, action_values, roi_predictor=roi_predictor)

    print("\nCQL Final Policy Distribution:")
    for a in range(5):
        print(f"  [{a}] {ACTION_NAMES[a]}: {cql_metrics['action_distribution'].get(a, 0):.2%}")

    print("\nDouble DQN Final Policy Distribution:")
    for a in range(5):
        print(f"  [{a}] {ACTION_NAMES[a]}: {dqn_metrics['action_distribution'].get(a, 0):.2%}")

    print("\nEnsemble CQL Final Policy Distribution:")
    for a in range(5):
        print(f"  [{a}] {ACTION_NAMES[a]}: {ensemble_metrics['action_distribution'].get(a, 0):.2%}")

    print("\n[7/8] Generating plots and report...")
    plot_learning_curves(cql_history, dqn_history, SAVE_DIR, ensemble_history=None)
    report = save_comparison_report(cql_history, dqn_history, cql_metrics, dqn_metrics, SAVE_DIR, sac_metrics=None, ensemble_metrics=ensemble_metrics)

    cql.save(os.path.join(SAVE_DIR, 'cql_model.d3'))
    dqn.save(os.path.join(SAVE_DIR, 'double_dqn_model.d3'))
    ensemble_cql.save(os.path.join(SAVE_DIR, 'ensemble_cql'))
    with open(os.path.join(SAVE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"All results saved to: {SAVE_DIR}")

    return cql, dqn, ensemble_cql, cql_history, dqn_history


if __name__ == "__main__":
    main()
