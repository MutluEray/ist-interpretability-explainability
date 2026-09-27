"""
Day 3 — Train the black-box model: gradient-boosted trees, via
scikit-learn's HistGradientBoostingClassifier.

(Originally XGBoost, then LightGBM — both hit the same macOS libomp/
OpenMP linking issue, since both are compiled libraries with an
external OpenMP dependency. HistGradientBoostingClassifier is the
same histogram-based GBT family but pure sklearn, no external compiled
dependency, so it sidesteps the problem entirely.)

Uses the same preprocessor as train_interpretable.py (fit fresh here,
on the same train split) so the black-box model sees identical inputs
to the interpretable models — required for the Day 4 comparison to be
meaningful. A light RandomizedSearchCV (not a full grid) picks
hyperparameters; this isn't the paper's contribution, so don't over-invest here.

Note: HistGradientBoostingClassifier has no native feature_importances_
(unlike XGBoost/LightGBM) — that's fine, the project doesn't need one.
The actual comparison is interpretable models' native importances
(train_interpretable.py) vs. SHAP/permutation importance on this
black-box model (explainability.py, Day 4).
"""

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import (
    TRAIN_CSV, TEST_CSV, BASELINE_FEATURES, TARGET_COL, RESULTS,
    RANDOM_SEED, build_preprocessor,
)

PARAM_DIST = {
    "max_iter": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, None],
    "max_leaf_nodes": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_samples_leaf": [5, 10, 20, 30],
    "l2_regularization": [0.0, 0.1, 1.0],
}


def train_gbt(X_train, y_train, n_iter=25):
    base_model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best CV AUROC: {search.best_score_:.3f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def evaluate(model, X_test, y_test, name="HistGradientBoosting") -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    print(f"{name}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
    return {"model": name, "auroc": auroc, "auprc": auprc}


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train_raw = train_df[BASELINE_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test_raw = test_df[BASELINE_FEATURES]
    y_test = test_df[TARGET_COL]

    # Fit a fresh preprocessor on this same train split — same
    # transformation logic as train_interpretable.py, fit independently
    # here so this script can be run standalone.
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    model = train_gbt(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RESULTS / "blackbox_model.joblib")
    pd.DataFrame([metrics]).to_csv(RESULTS / "blackbox_metrics.csv", index=False)

    print(f"\nSaved model and metrics to {RESULTS}")


if __name__ == "__main__":
    main()
