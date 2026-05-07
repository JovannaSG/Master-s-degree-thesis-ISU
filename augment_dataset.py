"""
Dataset Augmentation for Ad Campaign RL
========================================
Generates synthetic samples to expand the dataset from ~731 to ~3500+ samples
using domain-specific augmentation strategies.
"""

import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_PATH = r"test_datasets_for_RL/dtsts/processed_data.csv"
SAVE_PATH = r"pre_vkr/augmented_dataset_reduced.csv"  # New smaller dataset
N_COPIES_GAUSSIAN = 1  # Only 1 Gaussian copy per sample (reduced from 4)
NOISE_LEVEL = 0.08  # 8% standard deviation for continuous features
OVERSAMPLE_TOP_REWARDS = 1  # Disable oversampling (multiplier=1 means no duplication)
RANDOM_SEED = 42

# Continuous feature indices in observation array [CPC, CTR, CPC_change, Conversions, Position, Day, DeviceType, IsWeekend]
CONTINUOUS_INDICES = [0, 1, 2, 3, 4]  # CPC, CTR, CPC_change, Conversions, Position
# Indices to clip to valid ranges after perturbation
CLIP_RANGES = {
    0: (0, None),      # CPC >= 0
    1: (0, 100),       # CTR 0-100%
    2: (None, None),   # CPC_change can be negative
    3: (0, None),      # Conversions >= 0
    4: (1, 10),        # Position 1-10
}


def load_and_preprocess(path):
    """Load and preprocess the original dataset."""
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')

    # Convert numeric columns
    numeric_cols = ['CTR (%)', 'Конверсии', 'Клики', 'Показы', 'Расход (руб.)',
                    'Ср. цена клика (руб.)', 'Ср. позиция показов', 'Доход (руб.)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Filter to only active samples
    if 'Клики' in df.columns:
        mask = df['Клики'] > 0
        if 'Конверсии' in df.columns:
            mask = mask | (df['Конверсии'] > 0)
        df = df[mask].copy()
        df = df.reset_index(drop=True)

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


def compute_rewards(df):
    """Compute reward for filtering/oversampling (same logic as training script)."""
    df['CTR (%)'] = pd.to_numeric(df['CTR (%)'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Конверсии'] = pd.to_numeric(df['Конверсии'], errors='coerce').fillna(0)
    df['Клики'] = pd.to_numeric(df['Клики'], errors='coerce').fillna(0)

    has_clicks = (df['Клики'] > 0).astype(float)
    has_conversions = (df['Конверсии'] > 0).astype(float) * 2
    reward = has_clicks + has_conversions

    ctr = df['CTR (%)'].fillna(0) / 100
    reward = reward + ctr * 0.1

    # Normalize to [-1, 1]
    min_r = reward.min()
    max_r = reward.max()
    if max_r > min_r:
        df['reward'] = 2 * (reward - min_r) / (max_r - min_r) - 1
    else:
        df['reward'] = np.zeros(len(df))

    return df


def gaussian_perturbation(obs_array, noise_level=0.08):
    """Add Gaussian noise to continuous features of observation."""
    obs_new = obs_array.copy()

    for idx in CONTINUOUS_INDICES:
        feature_val = obs_new[idx]
        std = abs(feature_val) * noise_level + 0.01  # Ensure minimum std
        noise = np.random.normal(0, std)

        # For binary features (Day, DeviceType, IsWeekend), skip or add small noise
        if idx in [5, 6, 7]:  # Day, DeviceType, IsWeekend
            continue

        obs_new[idx] = feature_val + noise

        # Clip to valid range
        if idx in CLIP_RANGES:
            min_val, max_val = CLIP_RANGES[idx]
            if min_val is not None:
                obs_new[idx] = max(min_val, obs_new[idx])
            if max_val is not None:
                obs_new[idx] = min(max_val, obs_new[idx])

    return obs_new


def augment_with_gaussian(df, n_copies=4, noise_level=0.08):
    """Create augmented samples with Gaussian perturbation."""
    np.random.seed(RANDOM_SEED)
    augmented_rows = []

    for idx, row in df.iterrows():
        original_obs = row['observation_array']

        for copy_i in range(n_copies):
            new_row = row.copy()
            new_obs = gaussian_perturbation(original_obs, noise_level)
            new_row['observation'] = str(list(new_obs))
            new_row['is_augmented'] = True
            new_row['augmentation_type'] = f'gaussian_{copy_i}'
            augmented_rows.append(new_row)

    return pd.DataFrame(augmented_rows)


def oversample_high_rewards(df, multiplier=3):
    """Oversample top-performing samples (by reward)."""
    reward_threshold = df['reward'].quantile(0.8)  # Top 20%
    high_reward_samples = df[df['reward'] >= reward_threshold]

    oversampled = []
    for _ in range(multiplier - 1):  # -1 because original already counted
        oversampled.append(high_reward_samples.copy())

    if oversampled:
        return pd.concat([df] + oversampled, ignore_index=True)
    return df


def create_temporal_subsequences(df, min_len=3, max_len=10, n_subsequences=200):
    """Create new episodes from random subsequences of existing campaigns."""
    np.random.seed(RANDOM_SEED)
    subsequences = []

    # Group by campaign
    if '№ Группы' in df.columns:
        grouped = df.groupby('№ Группы')

        for _ in range(n_subsequences):
            # Pick random campaign
            campaign_id = np.random.choice(list(grouped.groups.keys()))
            campaign_df = grouped.get_group(campaign_id).copy()

            if len(campaign_df) < min_len:
                continue

            # Pick random subsequence
            start = np.random.randint(0, max(1, len(campaign_df) - min_len))
            length = np.random.randint(min_len, min(max_len + 1, len(campaign_df) - start + 1))
            end = start + length

            subseq = campaign_df.iloc[start:end].copy()
            subseq['is_augmented'] = True
            subseq['augmentation_type'] = 'temporal_subsequence'
            subsequences.append(subseq)

    return pd.concat(subsequences, ignore_index=True) if subsequences else pd.DataFrame()


def main():
    print("=" * 60)
    print("DATASET AUGMENTATION FOR RL")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading original dataset...")
    df = load_and_preprocess(DATA_PATH)
    print(f"Original filtered samples: {len(df)}")

    # Parse observations
    print("\n[2/5] Parsing observations...")
    df = parse_observations(df)

    # Compute rewards for stratification
    print("\n[3/5] Computing rewards...")
    df = compute_rewards(df)
    print(f"Reward range: [{df['reward'].min():.4f}, {df['reward'].max():.4f}]")

    # Add augmentation flag
    df['is_augmented'] = False
    df['augmentation_type'] = 'original'

    # Augmentation 1: Gaussian perturbation
    print(f"\n[4/5] Augmenting with Gaussian noise ({N_COPIES_GAUSSIAN} copies per sample)...")
    gaussian_aug = augment_with_gaussian(df, n_copies=N_COPIES_GAUSSIAN, noise_level=NOISE_LEVEL)
    print(f"Gaussian-augmented samples: {len(gaussian_aug)}")

    # Augmentation 2: Oversample high rewards
    print(f"\n[5/5] Oversampling high-reward samples (multiplier={OVERSAMPLE_TOP_REWARDS})...")
    df_with_oversample = oversample_high_rewards(df, multiplier=OVERSAMPLE_TOP_REWARDS)
    print(f"After oversampling: {len(df_with_oversample)} samples")

    # Augmentation 3: Temporal subsequences (DISABLED for more realistic data)
    subseq_df = pd.DataFrame()  # Empty - no temporal subsequences
    print("Temporal subsequences: DISABLED")

    # Combine only original + Gaussian (no oversampling, no temporal)
    print("\nCombining data (original + Gaussian only)...")
    all_data = pd.concat([df, gaussian_aug], ignore_index=True)
    all_data = all_data.reset_index(drop=True)

    # Remove observation_array column before saving (keep original 'observation' string)
    if 'observation_array' in all_data.columns:
        all_data = all_data.drop('observation_array', axis=1)

    # Save
    print(f"\nSaving augmented dataset to: {SAVE_PATH}")
    all_data.to_csv(SAVE_PATH, sep=';', encoding='utf-8-sig', index=False)

    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(all_data)}")
    print(f"Expansion factor: {len(all_data) / len(df):.1f}x")

    # Statistics
    print("\nAugmentation type distribution:")
    if 'augmentation_type' in all_data.columns:
        print(all_data['augmentation_type'].value_counts())

    print("\nReward statistics (augmented):")
    print(f"  Mean: {all_data['reward'].mean():.4f}")
    print(f"  Std: {all_data['reward'].std():.4f}")
    print(f"  Min: {all_data['reward'].min():.4f}")
    print(f"  Max: {all_data['reward'].max():.4f}")


if __name__ == "__main__":
    main()
