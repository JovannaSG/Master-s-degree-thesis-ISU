import pandas as pd
import numpy as np

INPUT_PATH = r"pre_vkr/augmented_dataset_merged.csv"
OUTPUT_PATH = r"pre_vkr/augmented_dataset_merged.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

KEEP_COLS = [
    'Дата',
    '№ Группы',
    'Показы',
    'Клики',
    'Расход (руб.)',
    'Конверсии',
    'Доход (руб.)',
    'CTR (%)',
    'Ср. цена клика (руб.)',
    'action',
    'observation',
    'is_augmented',
    'augmentation_type',
]

ZERO_CONV_KEEP_RATIO = 0.15


def main():
    print("=" * 60)
    print("CLEAN AND FILTER MERGED DATASET")
    print("=" * 60)

    print(f"\n[1/4] Loading merged dataset...")
    df = pd.read_csv(INPUT_PATH, sep=';', encoding='utf-8-sig', low_memory=False)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print(f"\n[2/4] Filtering: keeping {ZERO_CONV_KEEP_RATIO*100:.0f}% of zero-conversion rows...")
    df['Конверсии'] = pd.to_numeric(df['Конверсии'], errors='coerce').fillna(0)

    n_before = len(df)

    has_conv = df['Конверсии'] > 0
    no_conv = df['Конверсии'] == 0

    n_keep_no_conv = max(1, int(no_conv.sum() * ZERO_CONV_KEEP_RATIO))
    keep_indices = no_conv[no_conv].sample(n=n_keep_no_conv, random_state=RANDOM_SEED).index

    filtered = pd.concat([df[has_conv], df.loc[keep_indices]], ignore_index=True)
    n_after = len(filtered)

    print(f"  Before: {n_before} rows")
    print(f"  Kept all {has_conv.sum()} conversion rows")
    print(f"  Kept {n_keep_no_conv}/{no_conv.sum()} zero-conversion rows ({ZERO_CONV_KEEP_RATIO*100:.0f}%)")
    print(f"  After: {n_after} rows ({n_after/n_before*100:.1f}% retained)")

    print(f"\n[3/4] Removing {len(filtered.columns) - len(KEEP_COLS)} unnecessary columns...")
    available_keep = [c for c in KEEP_COLS if c in filtered.columns]
    removed = [c for c in filtered.columns if c not in available_keep]
    print(f"  Keeping: {available_keep}")
    print(f"  Removing: {removed}")

    cleaned = filtered[available_keep].copy()
    print(f"  New size: {cleaned.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print(f"\n[4/4] Saving to {OUTPUT_PATH}...")
    cleaned.to_csv(OUTPUT_PATH, sep=';', encoding='utf-8-sig', index=False)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Rows: {len(cleaned)}")
    print(f"  Columns: {len(cleaned.columns)}")
    print(f"  Zero-conversion: {(cleaned['Конверсии'] == 0).sum()} ({(cleaned['Конверсии'] == 0).mean()*100:.1f}%)")
    print(f"  Positive-conversion: {(cleaned['Конверсии'] > 0).sum()} ({(cleaned['Конверсии'] > 0).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
