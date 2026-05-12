import os, json, ast, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig

DATA_PATH = r"pre_vkr/augmented_dataset_merged.csv"
RESULTS_DIR = "./pre_vkr/test_ppo/results/cql_search"
TEST_SIZE = 0.2
N_STEPS = 30000
STEPS_PER_EPOCH = 1000

REWARD_WEIGHTS = {
    'w_ctr': 0.2, 'w_roi': 0.3, 'w_cpa': 0.2,
    'fine_high_spend_no_conv': 0.5, 'fine_increase_low_ctr': 0.3, 'fine_decrease_good_conv': 0.2,
}
ACTION_NAMES = {0: "Decrease bid", 1: "Slight decrease", 2: "Keep same",
                3: "Slight increase", 4: "Increase bid"}

ALPHA_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]
LEARNING_RATES = [1e-4, 3e-4]
GAMMA_VALUES = [0.95, 0.99]
BATCH_SIZE = 128


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


def compute_per_action_means(train_acts, train_rews):
    per_action_means = np.zeros(5)
    for a in range(5):
        mask = train_acts == a
        if mask.sum() > 0:
            per_action_means[a] = train_rews[mask].mean()
    return per_action_means


def train_and_eval_cql(train_obs, train_acts, train_rews, train_terms,
                       test_obs, per_action_means,
                       alpha, lr, gamma, batch_size=128,
                       n_steps=30000, n_steps_per_epoch=1000):
    train_dataset = MDPDataset(
        observations=train_obs, actions=train_acts,
        rewards=train_rews, terminals=train_terms)

    cql = DiscreteCQL(
        DiscreteCQLConfig(batch_size=batch_size, gamma=gamma,
                          learning_rate=lr, alpha=alpha),
        device='cpu', enable_ddp=False)
    cql.build_with_dataset(train_dataset)

    for epoch, metrics in cql.fitter(train_dataset, n_steps=n_steps,
                                      n_steps_per_epoch=n_steps_per_epoch,
                                      show_progress=False):
        pass

    test_pred = cql.predict(test_obs)
    test_am = per_action_means[test_pred].mean()

    train_pred = cql.predict(train_obs)
    train_am = per_action_means[train_pred].mean()

    train_dist = np.bincount(train_pred, minlength=5) / len(train_pred) * 100
    test_dist = np.bincount(test_pred, minlength=5) / len(test_pred) * 100

    return {
        'train_am': float(train_am),
        'test_am': float(test_am),
        'train_dist': train_dist.tolist(),
        'test_dist': test_dist.tolist(),
        'main_action_test': int(np.argmax(test_dist)),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_and_preprocess(DATA_PATH)
    df = compute_rewards(df, budget_factor=1.0)

    (full_obs, full_acts, full_rews, full_terms, group_ids) = \
        make_budgeted_data(df, budget_factor=1.0)

    train_mask, test_mask, train_set, test_set = episode_split(
        group_ids, train_size=1 - TEST_SIZE, random_state=42)

    train_obs = full_obs[train_mask]
    train_acts = full_acts[train_mask].ravel()
    train_rews = full_rews[train_mask].ravel()
    train_terms = full_terms[train_mask].ravel()

    test_obs = full_obs[test_mask]

    scaler = StandardScaler()
    train_obs_feat = scaler.fit_transform(train_obs[:, :8])
    train_obs_norm = np.column_stack([train_obs_feat, train_obs[:, 8:]])
    test_obs_feat = scaler.transform(test_obs[:, :8])
    test_obs_norm = np.column_stack([test_obs_feat, test_obs[:, 8:]])

    per_action_means = compute_per_action_means(train_acts, train_rews)
    dataset_baseline = float(per_action_means[train_acts].mean())
    print(f"Dataset baseline: {dataset_baseline:.4f}")
    print(f"Per-action means: {per_action_means}")

    results = []
    total = len(ALPHA_VALUES) * len(LEARNING_RATES) * len(GAMMA_VALUES)
    i = 0

    for alpha in ALPHA_VALUES:
        for lr in LEARNING_RATES:
            for gamma in GAMMA_VALUES:
                i += 1
                label = f"alpha={alpha}, lr={lr}, gamma={gamma}"
                print(f"\n[{i}/{total}] CQL {label} ...")
                t0 = time.time()

                try:
                    res = train_and_eval_cql(
                        train_obs_norm, train_acts, train_rews, train_terms,
                        test_obs_norm, per_action_means,
                        alpha=alpha, lr=lr, gamma=gamma,
                        batch_size=BATCH_SIZE,
                        n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)
                    res['alpha'] = alpha
                    res['lr'] = lr
                    res['gamma'] = gamma
                    res['time_sec'] = round(time.time() - t0, 1)
                    results.append(res)
                    delta = res['test_am'] - dataset_baseline
                    print(f"  Train AM={res['train_am']:.4f}, Test AM={res['test_am']:.4f} ({delta:+.4f} vs baseline)")
                    print(f"  Main action={ACTION_NAMES[res['main_action_test']]}, time={res['time_sec']:.0f}s")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        'alpha': alpha, 'lr': lr, 'gamma': gamma,
                        'error': str(e), 'test_am': -1e9
                    })

    print("\n" + "=" * 60)
    print("CQL PARAM SEARCH RESULTS (sorted by Test AM)")
    print("=" * 60)

    valid = [r for r in results if r.get('test_am', -1e9) > -1e8]
    valid.sort(key=lambda r: r['test_am'], reverse=True)

    print(f"{'Rank':<5} {'alpha':<6} {'lr':<10} {'gamma':<6} {'Train AM':>9} {'Test AM':>9} {'Delta':>8} {'Main action':<20} {'Time':>6}")
    print("-" * 80)
    for rank, r in enumerate(valid, 1):
        delta = r['test_am'] - dataset_baseline
        print(f"{rank:<5} {r['alpha']:<6} {r['lr']:<10} {r['gamma']:<6} {r['train_am']:>9.4f} {r['test_am']:>9.4f} {delta:>+8.4f} {ACTION_NAMES[r['main_action_test']]:<20} {r.get('time_sec', 0):>5.0f}s")

    best = valid[0]
    print(f"\nBest: alpha={best['alpha']}, lr={best['lr']}, gamma={best['gamma']}")
    print(f"  Test AM = {best['test_am']:.4f} (delta vs baseline: {best['test_am'] - dataset_baseline:+.4f})")

    out = {
        'dataset_baseline': dataset_baseline,
        'per_action_means': per_action_means.tolist(),
        'best_params': {k: best[k] for k in ['alpha', 'lr', 'gamma', 'test_am', 'train_am', 'main_action_test']},
        'results': valid,
    }

    with open(os.path.join(RESULTS_DIR, 'cql_search_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/cql_search_results.json")


if __name__ == '__main__':
    main()
