import os, json, ast, pickle, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DoubleDQN, DoubleDQNConfig, DiscreteBCQ, DiscreteBCQConfig, DiscreteCQL, DiscreteCQLConfig

DATA_PATH = r"pre_vkr/augmented_dataset_merged.csv"
SAVE_DIR = "./pre_vkr/test_ppo"
RESULTS_DIR = os.path.join(SAVE_DIR, "results")

TEST_SIZE = 0.2
N_STEPS = 50000
STEPS_PER_EPOCH = 1000

DQN_PARAMS = {'gamma': 0.99, 'batch_size': 256, 'learning_rate': 0.0003}
BCQ_PARAMS = {'gamma': 0.99, 'batch_size': 256, 'learning_rate': 6.25e-5, 'action_flexibility': 0.3, 'beta': 0.5}
CQL_PARAMS = {'gamma': 0.95, 'batch_size': 128, 'learning_rate': 0.0001, 'alpha': 0.2}
PPO_TIMESTEPS = 100000

REWARD_WEIGHTS = {
    'w_ctr': 0.2, 'w_roi': 0.3, 'w_cpa': 0.2,
    'fine_high_spend_no_conv': 0.5, 'fine_increase_low_ctr': 0.3, 'fine_decrease_good_conv': 0.2,
}
ACTION_NAMES = {0: "Decrease bid", 1: "Slight decrease", 2: "Keep same",
                3: "Slight increase", 4: "Increase bid"}
OBS_NAMES = ['CPC', 'CTR', 'CPC_change', 'Conversions', 'Position', 'Day', 'DeviceType', 'IsWeekend', 'BudgetRemaining']


def load_and_preprocess(path):
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    numeric_cols = [
        'CTR (%)', 'Конверсии', 'Клики', 'Показы', 'Расход (руб.)',
        'Ср. цена клика (руб.)', 'Ср. позиция показов', 'Доход (руб.)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    def parse_obs(obs_str):
        try:
            return np.array(ast.literal_eval(obs_str))
        except:
            return np.zeros(8)
    df['observation_array'] = df['observation'].apply(parse_obs)

    def normalize_act(action):
        if action < 50: return 0
        elif action < 70: return 1
        elif action < 80: return 2
        elif action < 100: return 3
        else: return 4
    df['action_normalized'] = df['action'].apply(normalize_act)

    df['CTR (%)'] = pd.to_numeric(df['CTR (%)'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Конверсии'] = pd.to_numeric(df['Конверсии'], errors='coerce').fillna(0)
    df['Клики'] = pd.to_numeric(df['Клики'], errors='coerce').fillna(0)
    df['Расход (руб.)'] = pd.to_numeric(df['Расход (руб.)'], errors='coerce').fillna(0)
    df['Доход (руб.)'] = pd.to_numeric(df['Доход (руб.)'], errors='coerce').fillna(0)
    return df


def compute_rewards(df, budget_factor=1.0):
    w = REWARD_WEIGHTS
    has_clicks = (df['Клики'] > 0).astype(float)
    has_conversions = (df['Конверсии'] > 0).astype(float) * 2
    reward = has_clicks + has_conversions

    ctr = df['CTR (%)'].fillna(0) / 100
    reward = reward + ctr * w['w_ctr']

    roi_mask = df['Расход (руб.)'] > 0
    roi = np.zeros(len(df))
    roi[roi_mask] = (df.loc[roi_mask, 'Доход (руб.)'] - df.loc[roi_mask, 'Расход (руб.)']) / df.loc[roi_mask, 'Расход (руб.)']
    roi = np.clip(roi, -1, 5)
    reward = reward + roi * w['w_roi']

    cpa_mask = df['Конверсии'] > 0
    cpa = np.zeros(len(df))
    cpa[cpa_mask] = df.loc[cpa_mask, 'Расход (руб.)'] / df.loc[cpa_mask, 'Конверсии']
    if cpa[cpa_mask].max() > cpa[cpa_mask].min():
        cpa_bonus = 1 - (cpa[cpa_mask] - cpa[cpa_mask].min()) / (cpa[cpa_mask].max() - cpa[cpa_mask].min())
        reward[cpa_mask] = reward[cpa_mask] + cpa_bonus * w['w_cpa']

    high_spend_no_conv = (df['Расход (руб.)'] > df['Расход (руб.)'].quantile(0.7)) & (df['Конверсии'] == 0)
    reward[high_spend_no_conv] -= w['fine_high_spend_no_conv']
    ctr_median = df['CTR (%)'].median()
    increase_bid_when_low_ctr = (df['action'] >= 100) & (df['CTR (%)'] < ctr_median)
    reward[increase_bid_when_low_ctr] -= w['fine_increase_low_ctr']
    decrease_when_converting = (df['action'] < 50) & (df['Конверсии'] > 0)
    reward[decrease_when_converting] -= w['fine_decrease_good_conv']

    min_r, max_r = reward.min(), reward.max()
    if max_r > min_r:
        df['reward_enhanced'] = 2 * (reward - min_r) / (max_r - min_r) - 1
    else:
        df['reward_enhanced'] = np.zeros(len(df))
    return df


def make_budgeted_data(df, budget_factor=1.0):
    groups = df['№ Группы'].values
    obs_array = np.stack(df['observation_array'].values)
    spends = df['Расход (руб.)'].values.astype(np.float32)
    acts = df['action_normalized'].values.astype(np.int32)
    rews = df['reward_enhanced'].values.astype(np.float32)

    obs_list, act_list, rew_list, term_list = [], [], [], []
    group_ids_list = []

    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        g_obs = obs_array[mask]
        g_spends = spends[mask]
        g_acts = acts[mask]
        g_rews = rews[mask]

        cumulative = np.cumsum(g_spends)
        total_spend = cumulative[-1]
        budget = total_spend * budget_factor

        within_budget = cumulative <= budget
        n_valid = max(int(within_budget.sum()), 1)

        budget_remaining = np.maximum(0, (budget - cumulative + g_spends) / max(budget, 1e-8))
        budget_remaining = np.clip(budget_remaining, 0, 1)

        augmented_obs = np.column_stack([g_obs[:n_valid], budget_remaining[:n_valid]])

        obs_list.append(augmented_obs)
        act_list.append(g_acts[:n_valid].astype(np.int32))
        rew_list.append(g_rews[:n_valid].astype(np.float32))

        terms = np.zeros(n_valid, dtype=np.float32)
        terms[-1] = 1.0
        term_list.append(terms)
        group_ids_list.append(np.full(n_valid, g, dtype=np.int64))

    return (np.concatenate(obs_list), np.concatenate(act_list),
            np.concatenate(rew_list), np.concatenate(term_list),
            np.concatenate(group_ids_list))


def episode_split(groups, train_size=0.8, random_state=42):
    unique = sorted(set(groups))
    n_train = int(len(unique) * train_size)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique)
    train_set = set(unique[:n_train])
    test_set = set(unique[n_train:])
    train_mask = np.array([g in train_set for g in groups])
    test_mask = np.array([g in test_set for g in groups])
    return train_mask, test_mask, train_set, test_set


class DatasetReplayEnv(gym.Env):
    def __init__(self, obs, acts, rews, terms, groups, group_set):
        super().__init__()
        self.obs = obs
        self.acts = acts
        self.rews = rews
        self.terms = terms
        self.groups = groups
        self.unique_episodes = sorted(group_set)

        self.n_features = obs.shape[1]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float64)

        self.episode_map = {}
        for g in self.unique_episodes:
            mask = self.groups == g
            indices = np.where(mask)[0]
            self.episode_map[g] = indices

        self.current_episode = None
        self.episode_indices = None
        self.pos = 0
        self.rng = np.random.RandomState(42)

    def _pick_episode(self):
        g = self.rng.choice(self.unique_episodes)
        self.current_episode = g
        self.episode_indices = self.episode_map[g]
        self.pos = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._pick_episode()
        return self.obs[self.episode_indices[self.pos]].copy(), {}

    def step(self, action):
        idx = self.episode_indices[self.pos]
        reward = float(self.rews[idx])
        terminated = bool(self.terms[idx])
        truncated = False
        info = {'dataset_action': int(self.acts[idx])}

        if terminated or self.pos >= len(self.episode_indices) - 1:
            next_obs = self.obs[self.episode_indices[0]].copy()
            self._pick_episode()
            return next_obs, reward, True, False, info

        self.pos += 1
        next_idx = self.episode_indices[self.pos]
        next_obs = self.obs[next_idx].copy()
        return next_obs, reward, terminated, truncated, info


def normalize_obs(obs, scaler=None):
    obs_feat = obs[:, :8]
    budget_feat = obs[:, 8:]
    if scaler is None:
        scaler = StandardScaler()
        obs_feat = scaler.fit_transform(obs_feat)
    else:
        obs_feat = scaler.transform(obs_feat)
    return np.column_stack([obs_feat, budget_feat]), scaler


def compute_per_action_means(train_acts, train_rews):
    per_action_means = np.zeros(5)
    for a in range(5):
        mask = train_acts == a
        if mask.sum() > 0:
            per_action_means[a] = train_rews[mask].mean()
    return per_action_means


def train_dqn(train_obs, train_acts, train_rews, train_terms,
              test_obs, test_orig_acts, test_orig_rews,
              per_action_means, n_steps=50000, n_steps_per_epoch=1000):
    train_dataset = MDPDataset(
        observations=train_obs, actions=train_acts,
        rewards=train_rews, terminals=train_terms)

    dqn = DoubleDQN(
        DoubleDQNConfig(batch_size=DQN_PARAMS['batch_size'],
                        gamma=DQN_PARAMS['gamma'],
                        learning_rate=DQN_PARAMS['learning_rate']),
        device='cpu', enable_ddp=False)
    dqn.build_with_dataset(train_dataset)

    print("\nTraining DQN...")
    for epoch, metrics in dqn.fitter(train_dataset, n_steps=n_steps,
                                      n_steps_per_epoch=n_steps_per_epoch,
                                      show_progress=True):
        pass

    test_pred = dqn.predict(test_obs)
    test_am = per_action_means[test_pred]
    dqn_per_action_return = test_am.mean()

    train_pred = dqn.predict(train_obs)
    train_am = per_action_means[train_pred]
    dqn_train_return = train_am.mean()

    return dqn, dqn_train_return, dqn_per_action_return, train_pred, test_pred


def train_bcq(train_obs, train_acts, train_rews, train_terms,
              test_obs, test_orig_acts, test_orig_rews,
              per_action_means, n_steps=50000, n_steps_per_epoch=1000):
    train_dataset = MDPDataset(
        observations=train_obs, actions=train_acts,
        rewards=train_rews, terminals=train_terms)

    bcq = DiscreteBCQ(
        DiscreteBCQConfig(batch_size=BCQ_PARAMS['batch_size'],
                          gamma=BCQ_PARAMS['gamma'],
                          learning_rate=BCQ_PARAMS['learning_rate'],
                          action_flexibility=BCQ_PARAMS['action_flexibility'],
                          beta=BCQ_PARAMS['beta']),
        device='cpu', enable_ddp=False)
    bcq.build_with_dataset(train_dataset)

    print("\nTraining DiscreteBCQ...")
    for epoch, metrics in bcq.fitter(train_dataset, n_steps=n_steps,
                                      n_steps_per_epoch=n_steps_per_epoch,
                                      show_progress=True):
        pass

    test_pred = bcq.predict(test_obs)
    test_am = per_action_means[test_pred]
    bcq_test_return = test_am.mean()

    train_pred = bcq.predict(train_obs)
    train_am = per_action_means[train_pred]
    bcq_train_return = train_am.mean()

    return bcq, bcq_train_return, bcq_test_return, train_pred, test_pred


def train_cql(train_obs, train_acts, train_rews, train_terms,
              test_obs, test_orig_acts, test_orig_rews,
              per_action_means, n_steps=50000, n_steps_per_epoch=1000):
    train_dataset = MDPDataset(
        observations=train_obs, actions=train_acts,
        rewards=train_rews, terminals=train_terms)

    cql = DiscreteCQL(
        DiscreteCQLConfig(batch_size=CQL_PARAMS['batch_size'],
                          gamma=CQL_PARAMS['gamma'],
                          learning_rate=CQL_PARAMS['learning_rate'],
                          alpha=CQL_PARAMS['alpha']),
        device='cpu', enable_ddp=False)
    cql.build_with_dataset(train_dataset)

    print("\nTraining DiscreteCQL...")
    for epoch, metrics in cql.fitter(train_dataset, n_steps=n_steps,
                                      n_steps_per_epoch=n_steps_per_epoch,
                                      show_progress=True):
        pass

    test_pred = cql.predict(test_obs)
    test_am = per_action_means[test_pred]
    cql_test_return = test_am.mean()

    train_pred = cql.predict(train_obs)
    train_am = per_action_means[train_pred]
    cql_train_return = train_am.mean()

    return cql, cql_train_return, cql_test_return, train_pred, test_pred


def evaluate_ppo_policy(model, obs, per_action_means):
    preds = model.predict(obs, deterministic=True)[0]
    am = per_action_means[preds]
    return am.mean(), preds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("DQN vs CQL vs BCQ vs PPO Comparison (Offline RL)")
    print("=" * 60)

    df = load_and_preprocess(DATA_PATH)
    df = compute_rewards(df, budget_factor=1.0)

    print("\nBuilding 9-feature observation (8 + budget_remaining)...")
    (full_obs, full_acts, full_rews, full_terms, group_ids) = \
        make_budgeted_data(df, budget_factor=1.0)

    print(f"  Total samples: {len(full_obs)}")
    print(f"  Total episodes: {len(set(group_ids))}")
    print(f"  Observation dim: {full_obs.shape[1]}")

    train_mask, test_mask, train_set, test_set = episode_split(
        group_ids, train_size=1 - TEST_SIZE, random_state=42)
    print(f"  Train episodes: {len(train_set)}, Test episodes: {len(test_set)}")

    train_obs = full_obs[train_mask]
    train_acts = full_acts[train_mask].ravel()
    train_rews = full_rews[train_mask].ravel()
    train_terms = full_terms[train_mask].ravel()
    train_groups = group_ids[train_mask]

    test_obs = full_obs[test_mask]
    test_acts = full_acts[test_mask].ravel()
    test_rews = full_rews[test_mask].ravel()
    test_terms = full_terms[test_mask].ravel()

    train_obs_norm, scaler = normalize_obs(train_obs)
    test_obs_norm, _ = normalize_obs(test_obs, scaler)

    per_action_means = compute_per_action_means(train_acts, train_rews)
    print(f"\nPer-action mean rewards (train):")
    for a in range(5):
        cnt = (train_acts == a).sum()
        print(f"  Action {a} ({ACTION_NAMES[a]}): mean={per_action_means[a]:.4f}, count={cnt}")

    dataset_baseline = per_action_means[train_acts].mean()
    print(f"\nDataset baseline (train): {dataset_baseline:.4f}")

    dqn_model, dqn_train_am, dqn_test_am, dqn_train_pred, dqn_test_pred = train_dqn(
        train_obs_norm, train_acts, train_rews, train_terms,
        test_obs_norm, test_acts, test_rews, per_action_means,
        n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)

    print(f"\nDQN Results:")
    print(f"  Train per-action mean return: {dqn_train_am:.4f}")
    print(f"  Test per-action mean return:  {dqn_test_am:.4f}")
    dqn_train_action_dist = np.bincount(dqn_train_pred, minlength=5) / len(dqn_train_pred) * 100
    dqn_test_action_dist = np.bincount(dqn_test_pred, minlength=5) / len(dqn_test_pred) * 100
    print(f"  Train action dist: {np.round(dqn_train_action_dist, 1)}")
    print(f"  Test action dist:  {np.round(dqn_test_action_dist, 1)}")

    bcq_model, bcq_train_am, bcq_test_am, bcq_train_pred, bcq_test_pred = train_bcq(
        train_obs_norm, train_acts, train_rews, train_terms,
        test_obs_norm, test_acts, test_rews, per_action_means,
        n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)

    print(f"\nDiscreteBCQ Results:")
    print(f"  Train per-action mean return: {bcq_train_am:.4f}")
    print(f"  Test per-action mean return:  {bcq_test_am:.4f}")
    bcq_train_action_dist = np.bincount(bcq_train_pred, minlength=5) / len(bcq_train_pred) * 100
    bcq_test_action_dist = np.bincount(bcq_test_pred, minlength=5) / len(bcq_test_pred) * 100
    print(f"  Train action dist:  {np.round(bcq_train_action_dist, 1)}")
    print(f"  Test action dist:  {np.round(bcq_test_action_dist, 1)}")

    cql_model, cql_train_am, cql_test_am, cql_train_pred, cql_test_pred = train_cql(
        train_obs_norm, train_acts, train_rews, train_terms,
        test_obs_norm, test_acts, test_rews, per_action_means,
        n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)

    print(f"\nDiscreteCQL Results:")
    print(f"  Train per-action mean return: {cql_train_am:.4f}")
    print(f"  Test per-action mean return:  {cql_test_am:.4f}")
    cql_train_action_dist = np.bincount(cql_train_pred, minlength=5) / len(cql_train_pred) * 100
    cql_test_action_dist = np.bincount(cql_test_pred, minlength=5) / len(cql_test_pred) * 100
    print(f"  Train action dist: {np.round(cql_train_action_dist, 1)}")
    print(f"  Test action dist:  {np.round(cql_test_action_dist, 1)}")

    print("\nTraining PPO (via SB3, dataset-replay env)...")
    train_env = DatasetReplayEnv(train_obs, train_acts, train_rews, train_terms, train_groups, train_set)

    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.callbacks import BaseCallback

    ppo = SB3PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=42,
    )

    ppo_train_rets = []

    class EvalCallback(BaseCallback):
        def __init__(self, eval_env, per_action_means, eval_freq=5000):
            super().__init__()
            self.eval_env = eval_env
            self.per_action_means = per_action_means
            self.eval_freq = eval_freq
            self.train_returns = []

        def _on_step(self):
            if self.num_timesteps % self.eval_freq == 0:
                ret, _ = evaluate_ppo_policy(
                    self.model, self.eval_env.obs, self.per_action_means)
                self.train_returns.append(ret)
                print(f"  PPO step {self.num_timesteps}: train_am_return={ret:.4f}")
            return True

    ppo.learn(
        total_timesteps=PPO_TIMESTEPS,
        callback=EvalCallback(train_env, per_action_means, eval_freq=5000),
        progress_bar=True,
    )

    ppo_train_am, ppo_train_pred = evaluate_ppo_policy(ppo, train_env.obs, per_action_means)
    ppo_test_am, ppo_test_pred = evaluate_ppo_policy(ppo, test_obs, per_action_means)

    print(f"\nPPO Results:")
    print(f"  Train per-action mean return: {ppo_train_am:.4f}")
    print(f"  Test per-action mean return:  {ppo_test_am:.4f}")
    ppo_train_action_dist = np.bincount(ppo_train_pred, minlength=5) / len(ppo_train_pred) * 100
    ppo_test_action_dist = np.bincount(ppo_test_pred, minlength=5) / len(ppo_test_pred) * 100
    print(f"  Train action dist: {np.round(ppo_train_action_dist, 1)}")
    print(f"  Test action dist:  {np.round(ppo_test_action_dist, 1)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algo':<10} {'Train AM':>10} {'Test AM':>10} {'Main action (test)':>20}")
    print("-" * 50)
    dqn_main_a = np.argmax(dqn_test_action_dist)
    bcq_main_a = np.argmax(bcq_test_action_dist)
    cql_main_a = np.argmax(cql_test_action_dist)
    ppo_main_a = np.argmax(ppo_test_action_dist)
    print(f"{'DQN':<10} {dqn_train_am:>10.4f} {dqn_test_am:>10.4f} {ACTION_NAMES[dqn_main_a]:>20}")
    print(f"{'CQL':<10} {cql_train_am:>10.4f} {cql_test_am:>10.4f} {ACTION_NAMES[cql_main_a]:>20}")
    print(f"{'BCQ':<10} {bcq_train_am:>10.4f} {bcq_test_am:>10.4f} {ACTION_NAMES[bcq_main_a]:>20}")
    print(f"{'PPO':<10} {ppo_train_am:>10.4f} {ppo_test_am:>10.4f} {ACTION_NAMES[ppo_main_a]:>20}")
    print(f"{'Dataset':<10} {dataset_baseline:>10.4f} {'--':>10}")

    results = {
        'per_action_means': {str(k): float(v) for k, v in enumerate(per_action_means)},
        'dataset_baseline': float(dataset_baseline),
        'dqn': {
            'train_am_return': float(dqn_train_am),
            'test_am_return': float(dqn_test_am),
            'train_action_dist': dqn_train_action_dist.tolist(),
            'test_action_dist': dqn_test_action_dist.tolist(),
        },
        'bcq': {
            'train_am_return': float(bcq_train_am),
            'test_am_return': float(bcq_test_am),
            'train_action_dist': bcq_train_action_dist.tolist(),
            'test_action_dist': bcq_test_action_dist.tolist(),
        },
        'cql': {
            'train_am_return': float(cql_train_am),
            'test_am_return': float(cql_test_am),
            'train_action_dist': cql_train_action_dist.tolist(),
            'test_action_dist': cql_test_action_dist.tolist(),
        },
        'ppo': {
            'train_am_return': float(ppo_train_am),
            'test_am_return': float(ppo_test_am),
            'train_action_dist': ppo_train_action_dist.tolist(),
            'test_action_dist': ppo_test_action_dist.tolist(),
        },
        'dataset_action_dist': (np.bincount(train_acts, minlength=5) / len(train_acts) * 100).tolist(),
    }

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    ax = axes[0]
    labels = list(ACTION_NAMES.values())
    x = np.arange(len(labels))
    width = 0.15
    dataset_pcts = results['dataset_action_dist']
    ax.bar(x - 2 * width, dataset_pcts, width, label='Dataset', color='gray', alpha=0.5)
    ax.bar(x - 1 * width, results['dqn']['test_action_dist'], width, label='DQN', color='red', alpha=0.7)
    ax.bar(x + 0 * width, results['cql']['test_action_dist'], width, label='CQL', color='purple', alpha=0.7)
    ax.bar(x + 1 * width, results['bcq']['test_action_dist'], width, label='BCQ', color='blue', alpha=0.7)
    ax.bar(x + 2 * width, results['ppo']['test_action_dist'], width, label='PPO', color='green', alpha=0.7)
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Action Distribution (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    metrics = ['Dataset', 'DQN', 'CQL', 'BCQ', 'PPO']
    values = [dataset_baseline, dqn_test_am, cql_test_am, bcq_test_am, ppo_test_am]
    colors = ['gray', 'red', 'purple', 'blue', 'green']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Per-Action Mean Return')
    ax.set_title('Test Set Comparison')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.axis('off')
    text = (
        f"Per-action mean rewards:\n"
            + "\n".join(f"  {k}: {v:.4f}" for k, v in enumerate(per_action_means))
        + f"\n\nDataset baseline: {dataset_baseline:.4f}\n\n"
        + f"DQN improvement: {dqn_test_am - dataset_baseline:+.4f}\n"
        + f"CQL improvement: {cql_test_am - dataset_baseline:+.4f}\n"
        + f"BCQ improvement: {bcq_test_am - dataset_baseline:+.4f}\n"
        + f"PPO improvement: {ppo_test_am - dataset_baseline:+.4f}"
    )
    ax.text(0, 0.5, text, transform=ax.transAxes, fontsize=9, verticalalignment='center',
            fontfamily='monospace')
    ax.set_title('Details')

    ax = axes[3]
    ax.axis('off')
    text2 = (
        f"Algorithm comparison:\n"
        + f"D = Dataset (baseline)\n\n"
        + f"DQN:  Train={dqn_train_am:.4f}  Test={dqn_test_am:.4f}\n"
        + f"       Main: {ACTION_NAMES[dqn_main_a]}\n"
        + f"CQL:  Train={cql_train_am:.4f}  Test={cql_test_am:.4f}\n"
        + f"       Main: {ACTION_NAMES[cql_main_a]}\n"
        + f"BCQ:  Train={bcq_train_am:.4f}  Test={bcq_test_am:.4f}\n"
        + f"       Main: {ACTION_NAMES[bcq_main_a]}\n"
        + f"PPO:  Train={ppo_train_am:.4f}  Test={ppo_test_am:.4f}\n"
        + f"       Main: {ACTION_NAMES[ppo_main_a]}"
    )
    ax.text(0, 0.5, text2, transform=ax.transAxes, fontsize=9, verticalalignment='center',
            fontfamily='monospace')
    ax.set_title('Summary')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'), dpi=150)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
