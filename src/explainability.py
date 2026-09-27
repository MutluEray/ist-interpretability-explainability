"""
Day 4 — Apply SHAP (TreeSHAP, exact) and permutation importance to the
black-box model (HistGradientBoostingClassifier), then quantitatively
compare rankings against the interpretable models' native feature
importances (Spearman rank correlation — kept simple on purpose).

Also saves the raw SHAP values array and the processed test matrix to
RESULTS/, since faithfulness.py (Day 5) and stability.py (Day 6) both
need to recompute or reuse SHAP values on the same model/data.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

from utils import (
    TEST_CSV, BASELINE_FEATURES, TARGET_COL, RESULTS, RANDOM_SEED,
    get_feature_names,
)


def compute_shap_importance(model, X, feature_names):
    """TreeSHAP on the fitted HistGradientBoostingClassifier model.
    shap.TreeExplainer supports sklearn's HistGradientBoostingClassifier
    natively. Returns a DataFrame of mean |SHAP value| per feature, plus
    the raw shap_values array (needed again in faithfulness.py and
    stability.py — save it)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Some sklearn tree ensembles return a list (one array per class)
    # from TreeExplainer; HistGradientBoostingClassifier binary output
    # is typically a single 2D array already. Handle both defensively.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    return importance, shap_values


def compute_permutation_importance(model, X, y, feature_names, n_repeats=20):
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=RANDOM_SEED, scoring="roc_auc"
    )
    return pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)


def compare_rankings(*importance_specs) -> pd.DataFrame:
    """importance_specs: sequence of (label, df, value_col) tuples,
    each df having a 'feature' column. Returns a square DataFrame of
    pairwise Spearman rank correlations between the given importance
    rankings, aligned on 'feature'."""
    labels = [label for label, _, _ in importance_specs]
    merged = None
    for label, df, value_col in importance_specs:
        sub = df[["feature", value_col]].rename(columns={value_col: label})
        merged = sub if merged is None else merged.merge(sub, on="feature", how="outer")

    corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, label_i in enumerate(labels):
        for label_j in labels:
            rho, _ = spearmanr(merged[label_i], merged[label_j])
            corr.loc[label_i, label_j] = rho
    return corr


def main():
    test_df = pd.read_csv(TEST_CSV)
    X_test_raw = test_df[BASELINE_FEATURES]
    y_test = test_df[TARGET_COL]

    preprocessor = joblib.load(RESULTS / "preprocessor.joblib")
    model = joblib.load(RESULTS / "blackbox_model.joblib")

    X_test = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor)

    shap_importance, shap_values = compute_shap_importance(model, X_test, feature_names)
    perm_importance = compute_permutation_importance(model, X_test, y_test, feature_names)

    print("\nTop 10 features by mean |SHAP value|:")
    print(shap_importance.head(10).to_string(index=False))
    print("\nTop 10 features by permutation importance:")
    print(perm_importance.head(10).to_string(index=False))

    logreg_importance = pd.read_csv(RESULTS / "logreg_importance.csv")
    tree_importance = pd.read_csv(RESULTS / "tree_importance.csv")

    rank_corr = compare_rankings(
        ("LogReg", logreg_importance, "abs_coefficient"),
        ("Tree", tree_importance, "importance"),
        ("SHAP", shap_importance, "mean_abs_shap"),
        ("Permutation", perm_importance, "perm_importance_mean"),
    )
    print("\nSpearman rank correlation between importance rankings:")
    print(rank_corr.round(3).to_string())

    RESULTS.mkdir(parents=True, exist_ok=True)
    shap_importance.to_csv(RESULTS / "shap_importance.csv", index=False)
    perm_importance.to_csv(RESULTS / "permutation_importance.csv", index=False)
    rank_corr.to_csv(RESULTS / "importance_rank_correlation.csv")
    np.save(RESULTS / "shap_values.npy", shap_values)
    np.save(RESULTS / "X_test_processed.npy", X_test)

    print(f"\nSaved SHAP/permutation importances, rank correlation, and "
          f"raw SHAP values to {RESULTS}")


if __name__ == "__main__":
    main()
