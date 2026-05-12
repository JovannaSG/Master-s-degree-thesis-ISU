import os, json, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_ppo_dqn as cpd

cpd.CQL_PARAMS = {'gamma': 0.99, 'batch_size': 128, 'learning_rate': 0.0001, 'alpha': 0.05}

BUDGET_FACTORS = [0.3, 0.5, 0.7, 1.0]
RESULTS_DIR = "./pre_vkr/test_ppo/results/budget"
TEST_SIZE = 0.2
N_STEPS = 50000
STEPS_PER_EPOCH = 1000
PPO_TIMESTEPS = 100000


def train_ppo_on_data(train_obs, train_acts, train_rews, train_terms,
                      train_groups, train_set, test_obs, per_action_means):
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.callbacks import BaseCallback

    train_env = cpd.DatasetReplayEnv(
        train_obs, train_acts, train_rews, train_terms,
        train_groups, train_set)

    ppo = SB3PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, verbose=0, seed=42,
    )

    class EvalCallback(BaseCallback):
        def __init__(self, eval_env, pam, eval_freq=5000):
            super().__init__()
            self.eval_env = eval_env
            self.per_action_means = pam
            self.eval_freq = eval_freq
            self.train_returns = []

        def _on_step(self):
            if self.num_timesteps % self.eval_freq == 0:
                ret, _ = cpd.evaluate_ppo_policy(
                    self.model, self.eval_env.obs, self.per_action_means)
                self.train_returns.append(ret)
                print(f"    PPO step {self.num_timesteps}: train_am={ret:.4f}")
            return True

    ppo.learn(
        total_timesteps=PPO_TIMESTEPS,
        callback=EvalCallback(train_env, per_action_means, eval_freq=5000),
        progress_bar=True,
    )

    train_am, train_pred = cpd.evaluate_ppo_policy(ppo, train_env.obs, per_action_means)
    test_am, test_pred = cpd.evaluate_ppo_policy(ppo, test_obs, per_action_means)

    train_dist = np.bincount(train_pred, minlength=5) / len(train_pred) * 100
    test_dist = np.bincount(test_pred, minlength=5) / len(test_pred) * 100

    return {'train_am': float(train_am), 'test_am': float(test_am),
            'train_dist': train_dist.tolist(), 'test_dist': test_dist.tolist(),
            'main_action_test': int(np.argmax(test_dist))}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = cpd.load_and_preprocess(cpd.DATA_PATH)

    all_results = []

    for bf in BUDGET_FACTORS:
        print(f"\n" + "=" * 60)
        print(f"  BUDGET FACTOR = {bf}")
        print("=" * 60)

        df_rew = cpd.compute_rewards(df.copy(), budget_factor=bf)

        (full_obs, full_acts, full_rews, full_terms, group_ids) = \
            cpd.make_budgeted_data(df_rew, budget_factor=bf)

        train_mask, test_mask, train_set, test_set = cpd.episode_split(
            group_ids, train_size=1 - TEST_SIZE, random_state=42)

        train_obs = full_obs[train_mask]
        train_acts = full_acts[train_mask].ravel()
        train_rews = full_rews[train_mask].ravel()
        train_terms = full_terms[train_mask].ravel()
        train_groups = group_ids[train_mask]

        test_obs = full_obs[test_mask]
        test_acts = full_acts[test_mask].ravel()
        test_rews = full_rews[test_mask].ravel()
        test_terms = full_terms[test_mask].ravel()

        train_obs_norm, scaler = cpd.normalize_obs(train_obs)
        test_obs_norm, _ = cpd.normalize_obs(test_obs, scaler)

        per_action_means = cpd.compute_per_action_means(train_acts, train_rews)
        dataset_baseline = float(per_action_means[train_acts].mean())

        print(f"  Samples: {len(full_obs)}, Episodes: {len(set(group_ids))}")
        print(f"  Train episodes: {len(train_set)}, Test: {len(test_set)}")
        print(f"  Per-action means: {per_action_means}")
        print(f"  Dataset baseline: {dataset_baseline:.4f}")

        factor_result = {
            'budget_factor': bf,
            'dataset_baseline': dataset_baseline,
            'per_action_means': {str(k): float(v) for k, v in enumerate(per_action_means)},
        }

        for algo_name, train_fn in [
            ('dqn', lambda: cpd.train_dqn(
                train_obs_norm, train_acts, train_rews, train_terms,
                test_obs_norm, test_acts, test_rews, per_action_means,
                n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)),
            ('cql', lambda: cpd.train_cql(
                train_obs_norm, train_acts, train_rews, train_terms,
                test_obs_norm, test_acts, test_rews, per_action_means,
                n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)),
            ('bcq', lambda: cpd.train_bcq(
                train_obs_norm, train_acts, train_rews, train_terms,
                test_obs_norm, test_acts, test_rews, per_action_means,
                n_steps=N_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH)),
        ]:
            t0 = time.time()
            print(f"\n  Training {algo_name.upper()} ...")
            model, train_am, test_am, train_pred, test_pred = train_fn()
            train_dist = np.bincount(train_pred, minlength=5) / len(train_pred) * 100
            test_dist = np.bincount(test_pred, minlength=5) / len(test_pred) * 100
            main_a = int(np.argmax(test_dist))
            delta = test_am - dataset_baseline
            print(f"  {algo_name.upper()}: Train={train_am:.4f}, Test={test_am:.4f} ({delta:+.4f}), "
                  f"Main={cpd.ACTION_NAMES[main_a]}, Time={time.time()-t0:.0f}s")
            factor_result[algo_name] = {
                'train_am': float(train_am), 'test_am': float(test_am),
                'train_dist': train_dist.tolist(), 'test_dist': test_dist.tolist(),
                'main_action_test': main_a,
            }

        print(f"\n  Training PPO ...")
        t0 = time.time()
        ppo_result = train_ppo_on_data(
            train_obs, train_acts, train_rews, train_terms,
            train_groups, train_set, test_obs, per_action_means)
        ppo_delta = ppo_result['test_am'] - dataset_baseline
        print(f"  PPO: Train={ppo_result['train_am']:.4f}, Test={ppo_result['test_am']:.4f} "
              f"({ppo_delta:+.4f}), Main={cpd.ACTION_NAMES[ppo_result['main_action_test']]}, "
              f"Time={time.time()-t0:.0f}s")
        factor_result['ppo'] = ppo_result

        factor_result['dataset_action_dist'] = (
            np.bincount(train_acts, minlength=5) / len(train_acts) * 100).tolist()

        all_results.append(factor_result)

        factor_dir = os.path.join(RESULTS_DIR, f"factor_{bf}")
        os.makedirs(factor_dir, exist_ok=True)
        with open(os.path.join(factor_dir, 'results.json'), 'w') as f:
            json.dump(factor_result, f, indent=2)
        print(f"  Saved to {factor_dir}/")

    print("\n" + "=" * 60)
    print("  BUDGET COMPARISON SUMMARY")
    print("=" * 60)

    print(f"{'Factor':<8} {'Baseline':>9} {'DQN Test':>9} {'CQL Test':>9} {'BCQ Test':>9} {'PPO Test':>9}")
    print("-" * 55)
    for r in all_results:
        bf = r['budget_factor']
        bl = r['dataset_baseline']
        dqn = r['dqn']['test_am']
        cql = r['cql']['test_am']
        bcq = r['bcq']['test_am']
        ppo = r['ppo']['test_am']
        print(f"{bf:<8} {bl:>9.4f} {dqn:>9.4f} {cql:>9.4f} {bcq:>9.4f} {ppo:>9.4f}")

    with open(os.path.join(RESULTS_DIR, 'budget_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    factors = [r['budget_factor'] for r in all_results]
    x = np.arange(len(factors))
    width = 0.17
    ax.bar(x - 1.5 * width, [r['dataset_baseline'] for r in all_results], width,
           label='Dataset', color='gray', alpha=0.5)
    ax.bar(x - 0.5 * width, [r['dqn']['test_am'] for r in all_results], width,
           label='DQN', color='red', alpha=0.7)
    ax.bar(x + 0.5 * width, [r['cql']['test_am'] for r in all_results], width,
           label='CQL', color='purple', alpha=0.7)
    ax.bar(x + 1.5 * width, [r['bcq']['test_am'] for r in all_results], width,
           label='BCQ', color='blue', alpha=0.7)
    ax.bar(x + 2.5 * width, [r['ppo']['test_am'] for r in all_results], width,
           label='PPO', color='green', alpha=0.7)
    ax.set_xlabel('Budget Factor')
    ax.set_ylabel('Test Per-Action Mean Return')
    ax.set_title('Algorithm Comparison Across Budget Factors')
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in factors])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axis('off')
    lines = ["Per-Algorithm Test AM by Budget Factor:\n"]
    lines.append(f"{'Factor':>7} {'DQN':>8} {'CQL':>8} {'BCQ':>8} {'PPO':>8}")
    lines.append("-" * 43)
    for r in all_results:
        bf = r['budget_factor']
        dqn_delta = r['dqn']['test_am'] - r['dataset_baseline']
        cql_delta = r['cql']['test_am'] - r['dataset_baseline']
        bcq_delta = r['bcq']['test_am'] - r['dataset_baseline']
        ppo_delta = r['ppo']['test_am'] - r['dataset_baseline']
        lines.append(f"{bf:>7.1f} {dqn_delta:>+8.4f} {cql_delta:>+8.4f} {bcq_delta:>+8.4f} {ppo_delta:>+8.4f}")
    lines.append(f"\nBest DQN factor: {max(all_results, key=lambda r: r['dqn']['test_am'])['budget_factor']}")
    lines.append(f"Best CQL factor: {max(all_results, key=lambda r: r['cql']['test_am'])['budget_factor']}")
    lines.append(f"Best BCQ factor: {max(all_results, key=lambda r: r['bcq']['test_am'])['budget_factor']}")
    lines.append(f"Best PPO factor: {max(all_results, key=lambda r: r['ppo']['test_am'])['budget_factor']}")
    ax.text(0, 0.5, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace')
    ax.set_title('Delta vs Baseline')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'budget_comparison.png'), dpi=150)
    print(f"\nSummary saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
