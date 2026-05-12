import os, json, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_ppo_dqn as cpd

RESULTS_DIR = "./pre_vkr/test_ppo/results/bcq_search"
TEST_SIZE = 0.2
N_STEPS = 30000
STEPS_PER_EPOCH = 1000

ACTION_FLEXIBILITY = [0.1, 0.3, 0.5, 0.7, 1.0]
BETA_VALUES = [0.3, 0.5, 0.7, 1.0]
LEARNING_RATES = [6.25e-5, 3e-4]

BATCH_SIZE = 256
GAMMA = 0.99


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = cpd.load_and_preprocess(cpd.DATA_PATH)
    df = cpd.compute_rewards(df, budget_factor=1.0)

    (full_obs, full_acts, full_rews, full_terms, group_ids) = \
        cpd.make_budgeted_data(df, budget_factor=1.0)

    train_mask, test_mask, train_set, test_set = cpd.episode_split(
        group_ids, train_size=1 - TEST_SIZE, random_state=42)

    train_obs = full_obs[train_mask]
    train_acts = full_acts[train_mask].ravel()
    train_rews = full_rews[train_mask].ravel()
    train_terms = full_terms[train_mask].ravel()

    test_obs = full_obs[test_mask]

    train_obs_norm, scaler = cpd.normalize_obs(train_obs)
    test_obs_norm, _ = cpd.normalize_obs(test_obs, scaler)

    per_action_means = cpd.compute_per_action_means(train_acts, train_rews)
    dataset_baseline = float(per_action_means[train_acts].mean())

    print(f"Dataset baseline: {dataset_baseline:.4f}")
    print(f"Per-action means: {per_action_means}")
    print(f"Train samples: {len(train_obs)}, Test: {len(test_obs)}")

    results = []
    total = len(ACTION_FLEXIBILITY) * len(BETA_VALUES) * len(LEARNING_RATES)
    i = 0

    for af in ACTION_FLEXIBILITY:
        for beta in BETA_VALUES:
            for lr in LEARNING_RATES:
                i += 1
                label = f"af={af}, beta={beta}, lr={lr}"
                print(f"\n[{i}/{total}] BCQ {label} ...")
                t0 = time.time()

                try:
                    from d3rlpy.datasets import MDPDataset
                    from d3rlpy.algos import DiscreteBCQ, DiscreteBCQConfig

                    train_dataset = MDPDataset(
                        observations=train_obs_norm, actions=train_acts,
                        rewards=train_rews, terminals=train_terms)

                    bcq = DiscreteBCQ(
                        DiscreteBCQConfig(
                            batch_size=BATCH_SIZE, gamma=GAMMA,
                            learning_rate=lr,
                            action_flexibility=af, beta=beta),
                        device='cpu', enable_ddp=False)
                    bcq.build_with_dataset(train_dataset)

                    for epoch, metrics in bcq.fitter(
                            train_dataset, n_steps=N_STEPS,
                            n_steps_per_epoch=STEPS_PER_EPOCH,
                            show_progress=False):
                        pass

                    test_pred = bcq.predict(test_obs_norm)
                    test_am = float(per_action_means[test_pred].mean())

                    train_pred = bcq.predict(train_obs_norm)
                    train_am = float(per_action_means[train_pred].mean())

                    train_dist = np.bincount(train_pred, minlength=5) / len(train_pred) * 100
                    test_dist = np.bincount(test_pred, minlength=5) / len(test_pred) * 100
                    main_a = int(np.argmax(test_dist))
                    delta = test_am - dataset_baseline

                    print(f"  Train AM={train_am:.4f}, Test AM={test_am:.4f} ({delta:+.4f}), "
                          f"Main={cpd.ACTION_NAMES[main_a]}, Time={time.time()-t0:.0f}s")

                    results.append({
                        'action_flexibility': af, 'beta': beta, 'lr': lr,
                        'train_am': train_am, 'test_am': test_am,
                        'train_dist': train_dist.tolist(), 'test_dist': test_dist.tolist(),
                        'main_action_test': main_a,
                        'delta_vs_baseline': round(delta, 4),
                        'time_sec': round(time.time() - t0, 1),
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        'action_flexibility': af, 'beta': beta, 'lr': lr,
                        'error': str(e), 'test_am': -1e9,
                    })

    valid = [r for r in results if r.get('test_am', -1e9) > -1e8]
    valid.sort(key=lambda r: r['test_am'], reverse=True)

    print("\n" + "=" * 65)
    print("BCQ PARAM SEARCH RESULTS (sorted by Test AM)")
    print("=" * 65)
    print(f"{'Rank':<5} {'af':<5} {'beta':<6} {'lr':<10} {'Train AM':>9} {'Test AM':>9} "
          f"{'Delta':>8} {'Main action':<18} {'Time':>6}")
    print("-" * 75)
    for rank, r in enumerate(valid, 1):
        delta = r['test_am'] - dataset_baseline
        print(f"{rank:<5} {r['action_flexibility']:<5} {r['beta']:<6} {r['lr']:<10} "
              f"{r['train_am']:>9.4f} {r['test_am']:>9.4f} {delta:>+8.4f} "
              f"{cpd.ACTION_NAMES.get(r['main_action_test'], '?'):<18} {r.get('time_sec', 0):>5.0f}s")

    best = valid[0]
    print(f"\nBest: af={best['action_flexibility']}, beta={best['beta']}, lr={best['lr']}")
    print(f"  Test AM = {best['test_am']:.4f} (delta: {best['test_am'] - dataset_baseline:+.4f})")

    out = {
        'dataset_baseline': dataset_baseline,
        'per_action_means': per_action_means.tolist(),
        'bcq_default_params': {
            'action_flexibility': 0.3, 'beta': 0.5, 'lr': 6.25e-5,
        },
        'best_params': {
            'action_flexibility': best['action_flexibility'],
            'beta': best['beta'], 'lr': best['lr'],
            'test_am': best['test_am'], 'train_am': best['train_am'],
            'main_action_test': best['main_action_test'],
        },
        'results': valid,
    }
    with open(os.path.join(RESULTS_DIR, 'bcq_search_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/bcq_search_results.json")


if __name__ == '__main__':
    main()
