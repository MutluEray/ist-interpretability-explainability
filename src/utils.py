"""Shared paths, constants, and small helpers used across the pipeline."""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

RAW_CSV = DATA_RAW / "IST_corrected.csv"  # rename here if your file differs
RAW_CSV_ENCODING = "latin1"  # source file predates UTF-8-only exports; not valid UTF-8
CLEAN_CSV = DATA_PROCESSED / "ist_clean.csv"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

# --- Feature / target definitions --------------------------------------
# Baseline (randomisation) variables only — see README for rationale.
# COUNTRY is deliberately excluded: it's a proxy for site/healthcare-system
# variation, not patient-level clinical risk, and several countries have
# under 100 (down to single-digit) patients, which produces unstable,
# separation-driven logistic regression coefficients unrelated to any
# real clinical signal (confirmed empirically — see data_cleaning.py's
# COUNTRY sample-size check).
BASELINE_FEATURES = [
    "RDELAY", "RCONSC", "SEX", "AGE", "RSLEEP", "RATRIAL", "RCT",
    "RVISINF", "RHEP24", "RASP3", "RSBP",
    "RDEF1", "RDEF2", "RDEF3", "RDEF4", "RDEF5", "RDEF6", "RDEF7", "RDEF8",
    "STYPE",
]

# Explicitly excluded, and why (kept here so it's visible in one place,
# not just buried in a script):
#   RXASP, RXHEP        -> randomized treatment allocation, not baseline state
#   EXPDD, EXPD6, EXPD14 -> pre-computed prognostic-score probabilities (leakage)
#   all 14-day/discharge fields -> post-randomisation, not baseline

TARGET_RAW_COL = "OCCODE"
TARGET_COL = "target_death_dependency"  # binarized column created in target_definition.py

# Split for the shared preprocessing pipeline (see build_preprocessor below)
NUMERIC_FEATURES = ["RDELAY", "AGE", "RSBP"]
CATEGORICAL_FEATURES = [f for f in BASELINE_FEATURES if f not in NUMERIC_FEATURES]

RANDOM_SEED = 42

# --- Small helpers -------------------------------------------------------
def ensure_dirs():
    for d in (DATA_PROCESSED, FIGURES, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


def build_preprocessor():
    """Shared preprocessing: standardize numeric features, one-hot
    encode categorical ones. Used identically by train_interpretable.py
    and train_blackbox.py so the two models are compared on the same
    inputs. handle_unknown='ignore' guards against a category showing
    up in test that wasn't seen in train (rare here, but cheap to be safe).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor) -> list:
    """Readable post-encoding feature names (e.g. 'cat__SEX_M'),
    needed to map SHAP/importance values back to real feature names."""
    return list(preprocessor.get_feature_names_out())
