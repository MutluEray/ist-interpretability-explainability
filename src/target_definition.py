"""
Day 1 — Build the binary target from OCCODE, restrict to
BASELINE_FEATURES, drop rows with missing outcome, and write a
stratified train/test split.

OCCODE: 1=dead, 2=dependent, 3=not recovered, 4=recovered, 0/9=missing.
Target: death/dependency (OCCODE in {1,2}) vs. not (OCCODE in {3,4}).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    CLEAN_CSV, TRAIN_CSV, TEST_CSV,
    BASELINE_FEATURES, TARGET_RAW_COL, TARGET_COL, RANDOM_SEED,
)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[TARGET_RAW_COL].isin([1, 2, 3, 4])].copy()
    df[TARGET_COL] = df[TARGET_RAW_COL].isin([1, 2]).astype(int)
    print(f"Target balance:\n{df[TARGET_COL].value_counts(normalize=True)}")
    return df


def main():
    df = pd.read_csv(CLEAN_CSV)
    df = build_target(df)

    cols = BASELINE_FEATURES + [TARGET_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found after cleaning: {missing}")
    df = df[cols]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df[TARGET_COL]
    )
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Wrote {len(train_df)} train / {len(test_df)} test rows.")


if __name__ == "__main__":
    main()
