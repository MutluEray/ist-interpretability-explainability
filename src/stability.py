"""
Day 6 — Stability of post-hoc (SHAP) explanations under bootstrap
resampling of the training set.

Design choice: the preprocessor (one-hot categories, scaler) stays
fixed — already fit on the full original training set — and only the
already-processed training matrix is bootstrap-resampled before
refitting the black-box model. This isolates "does the explanation
change because of which patients happened to be in the training
sample" from "does the encoding scheme change," which would happen if
a bootstrap resample dropped a rare category and shifted which
one-hot columns exist. The evaluation sample (X_explain) is the same
300 test rows used in faithfulness.py, held fixed across all bootstrap
resamples, so what varies is purely the training data the model saw —
not the data being explained.

Model refits reuse the best hyperparameters already found in Day 3
(no repeated hyperparameter search — that would make 50 resamples far
too slow for a 1-week budget and isn't the point of this test).
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import (
    TRAIN_CSV, TEST_CSV, BASELINE_FEATURES, TARGET_COL, RESULTS, FIGURES,
    RANDOM_SEED, get_feature_names,
)

N_BOOT = 50
SAMPLE_SIZE = 300  # must match faithfulness.py's SAMPLE_SIZE for the same eval rows


def bootstrap_shap_importances(X_train: np.ndarray, y_train: np.ndarray,
                                X_explain: np.ndarray, best_params: dict,
                                n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> np.ndarray:
    """Refit the black-box model on n_boot bootstrap resamples of the
    training data (fixed hyperparameters), recompute TreeSHAP mean
    |value| per feature each time. Returns an (n_boot, n_features)
    array — one importance vector per resample."""
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    importances = []
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X_train[idx], y_train[idx]
        model = HistGradientBoostingClassifier(**{**best_params, "random_state": seed + i})
        model.fit(Xb, yb)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        importances.append(np.abs(shap_values).mean(axis=0))

        if (i + 1) % 10 == 0:
            print(f"  bootstrap {i + 1}/{n_boot} done")
    return np.array(importances)


def summarize_stability(importances: np.ndarray, feature_names: list):
    """Per-feature mean/std importance across resamples, plus a single
    overall stability score: mean pairwise Spearman correlation
    between every pair of bootstrap importance rankings. Close to 1.0
    = the ranking barely changes across resamples (stable); closer to
    0 = the ranking is highly sensitive to which patients were
    sampled (unstable)."""
    means = importances.mean(axis=0)
    stds = importances.std(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": means,
        "std_importance": stds,
    }).sort_values("mean_importance", ascending=False)

    n_boot = importances.shape[0]
    corrs = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            rho, _ = spearmanr(importances[i], importances[j])
            corrs.append(rho)
    mean_pairwise_corr = float(np.mean(corrs))
    return df, mean_pairwise_corr


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    preprocessor = joblib.load(RESULTS / "preprocessor.joblib")
    blackbox_model = joblib.load(RESULTS / "blackbox_model.joblib")
    feature_names = get_feature_names(preprocessor)
    best_params = blackbox_model.get_params()

    X_train = preprocessor.transform(train_df[BASELINE_FEATURES])
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = preprocessor.transform(test_df[BASELINE_FEATURES])

    # Same sampling logic as faithfulness.py, so this is the identical
    # 300-row evaluation set — only the training data varies here.
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(X_test.shape[0], size=min(SAMPLE_SIZE, X_test.shape[0]), replace=False)
    X_explain = X_test[sample_idx]

    print(f"Running {N_BOOT} bootstrap resamples...")
    importances = bootstrap_shap_importances(X_train, y_train, X_explain, best_params, n_boot=N_BOOT)

    stability_df, mean_pairwise_corr = summarize_stability(importances, feature_names)
    print(f"\nMean pairwise Spearman correlation across {N_BOOT} bootstrap "
          f"rankings: {mean_pairwise_corr:.3f}")
    print("(closer to 1.0 = stable ranking; closer to 0 = unstable)")
    print("\nTop 10 features by mean importance across bootstraps (with std):")
    print(stability_df.head(10).to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "bootstrap_shap_importances.npy", importances)
    stability_df.to_csv(RESULTS / "stability_summary.csv", index=False)
    pd.DataFrame([{"mean_pairwise_spearman": mean_pairwise_corr}]).to_csv(
        RESULTS / "stability_score.csv", index=False
    )

    # Draft plot: boxplot of importance distribution across bootstraps,
    # top 15 features by mean importance, sorted.
    FIGURES.mkdir(parents=True, exist_ok=True)
    top15 = stability_df.head(15)
    top15_idx = [feature_names.index(f) for f in top15["feature"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [importances[:, i] for i in top15_idx],
        vert=False,
        labels=top15["feature"],
    )
    ax.set_xlabel("Mean |SHAP value| across bootstrap resamples")
    ax.set_title(f"Stability of top 15 features ({N_BOOT} bootstrap resamples)")
    fig.tight_layout()
    fig.savefig(FIGURES / "stability_draft.png", dpi=150)

    print(f"\nSaved bootstrap importances, stability summary, and a draft plot "
          f"to {RESULTS} / {FIGURES}")


if __name__ == "__main__":
    main()
