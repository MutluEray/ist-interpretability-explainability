"""
Day 1 — Load the raw IST CSV, recode missingness, sanity-check shape,
and write a cleaned CSV to data/processed/.

The IST fields use Y/N/U/C-style codes rather than blank cells for
missing/unassessable data (e.g. RDEF1..RDEF8 use "C" for "can't assess").
Don't rely on pandas' default NaN detection — decide explicitly, per
column, what counts as missing and how it should be encoded, and write
that decision down as a comment next to the code that makes it.
"""

import pandas as pd

from utils import RAW_CSV, RAW_CSV_ENCODING, CLEAN_CSV, ensure_dirs


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding=RAW_CSV_ENCODING)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {RAW_CSV.name}")
    # Sanity check against the published dataset description (19,435 patients).
    assert df.shape[0] > 0, "No rows loaded — check RAW_CSV path in utils.py"
    return df


def check_rconsc_and_country(df: pd.DataFrame):
    """RCONSC natively uses 'U' to mean unconscious (F/D/U), which
    collides with 'U' as our fill value for missing data elsewhere.
    Check here, on the RAW column, whether RCONSC has any real nulls
    that would get wrongly merged into the 'unconscious' category by
    recode_missing(). Also print COUNTRY sample sizes — small-n
    countries risk quasi-separation blowing up logistic regression
    coefficients for reasons that have nothing to do with the clinical
    signal.
    """
    print("\n--- RCONSC raw check (before recoding) ---")
    n_null = df["RCONSC"].isna().sum()
    print(f"RCONSC null count: {n_null}  (should be 0 for 'U' to safely mean 'unconscious')")
    print(df["RCONSC"].value_counts(dropna=False))

    print("\n--- COUNTRY sample sizes (quasi-separation risk check) ---")
    counts = df["COUNTRY"].value_counts(dropna=False)
    print(counts)
    small = counts[counts < 100]
    if len(small) > 0:
        print(f"\nWARNING: {len(small)} countries have fewer than 100 patients — "
              f"logistic regression coefficients for these are likely unstable:")
        print(small)


def recode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Recode Y/N/U/C-style codes to a consistent scheme for the
    baseline features. Missing/blank values become an explicit 'U'
    (unknown) category for Y/N/C fields (informative missingness —
    e.g. RATRIAL/RASP3 are simply not coded for the 984 pilot-phase
    patients) rather than being silently dropped or imputed here.
    Numeric fields are coerced to numeric, with any conversion
    failures reported rather than silently swallowed.
    """
    df = df.copy()

    yn_cols = ["RSLEEP", "RATRIAL", "RCT", "RVISINF", "RHEP24", "RASP3"]
    def_cols = [f"RDEF{i}" for i in range(1, 9)]
    cat_cols = ["RCONSC", "SEX", "STYPE", "COUNTRY"]

    for col in yn_cols + def_cols + cat_cols:
        s = df[col]
        # Check for missing on the ORIGINAL values before any string
        # conversion — round-tripping through str() first turns real
        # NaN into the literal string "nan", which is fragile to match.
        is_missing = s.isna() | (s.astype(str).str.strip() == "")
        s = s.astype(str).str.strip().str.upper()
        s = s.mask(is_missing, "U")
        df[col] = s

    num_cols = ["RDELAY", "AGE", "RSBP"]
    for col in num_cols:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna().sum()
        if after_na > before_na:
            print(f"WARNING: {after_na - before_na} values in {col} "
                  f"failed numeric conversion and became NaN")

    return df


def sanity_check(df: pd.DataFrame):
    """Print category counts for every recoded column so you can
    eyeball that only the documented codes (Y/N/U, or Y/N/C/U for the
    deficit fields) appear, before this feeds into target_definition.py.
    """
    yn_cols = ["RSLEEP", "RATRIAL", "RCT", "RVISINF", "RHEP24", "RASP3"]
    def_cols = [f"RDEF{i}" for i in range(1, 9)]
    print("\n--- Sanity check: category counts after recoding ---")
    for col in yn_cols + def_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))


def main():
    ensure_dirs()
    df = load_raw()
    check_rconsc_and_country(df)
    df = recode_missing(df)
    sanity_check(df)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nWrote cleaned data to {CLEAN_CSV}")


if __name__ == "__main__":
    main()
