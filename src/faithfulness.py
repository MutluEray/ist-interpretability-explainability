"""
Day 5 — Faithfulness evaluation via deletion/insertion curves.

Feature granularity note: "features" here means the one-hot encoded
columns (e.g. cat__RCONSC_F, cat__RCONSC_U), consistent with how
LogReg coefficients, tree importance, SHAP, and permutation importance
were all reported in explainability.py. Masking is done per encoded
column, not per original clinical variable — worth a sentence in the
write-up so the granularity choice is explicit, not implicit.

Deletion: starting from real test data, progressively replace the
top-k most important features (ranked by SHAP mean |value|) with a
baseline (training-set median for that column), tracking how the
model's mean predicted probability moves. A faithful explanation
should show a sharp early drop.

Insertion: the mirror image — start from an all-baseline input and
progressively insert the top-k features back in, tracking how fast
the prediction recovers toward the real value.

Run on a sample of test rows (not all ~3,900) — averaging over a
few hundred is enough and keeps this fast.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    TRAIN_CSV, TEST_CSV, BASELINE_FEATURES, RESULTS, FIGURES,
    RANDOM_SEED, get_feature_names,
)

SAMPLE_SIZE = 300


def get_baseline_values(X_train: np.ndarray) -> np.ndarray:
    """Per-column median of the processed training matrix. For
    standardized numeric columns this sits near 0 (the scaled mean);
    for one-hot columns it's typically 0 (absence) unless that
    category is the majority class, in which case 1 — both are
    reasonable 'typical patient' baselines for masking."""
    return np.median(X_train, axis=0)


def deletion_curve(model, X_sample: np.ndarray, feature_order: list, baseline: np.ndarray) -> np.ndarray:
    """Mean predicted probability as features are masked to baseline,
    one at a time, in feature_order (most important first). Returns
    an array of length len(feature_order)+1; index 0 is the
    unmasked/original mean prediction."""
    X_work = X_sample.copy()
    curve = [model.predict_proba(X_work)[:, 1].mean()]
    for idx in feature_order:
        X_work[:, idx] = baseline[idx]
        curve.append(model.predict_proba(X_work)[:, 1].mean())
    return np.array(curve)


def insertion_curve(model, X_sample: np.ndarray, feature_order: list, baseline: np.ndarray) -> np.ndarray:
    """Mirror of deletion_curve: start from an all-baseline matrix,
    insert real feature values in feature_order (most important
    first). Returns an array of length len(feature_order)+1; index 0
    is the all-baseline mean prediction."""
    n_samples = X_sample.shape[0]
    X_work = np.tile(baseline, (n_samples, 1))
    curve = [model.predict_proba(X_work)[:, 1].mean()]
    for idx in feature_order:
        X_work[:, idx] = X_sample[:, idx]
        curve.append(model.predict_proba(X_work)[:, 1].mean())
    return np.array(curve)


def random_order_curves(model, X_sample: np.ndarray, n_features: int, baseline: np.ndarray,
                         n_repeats: int = 10, seed: int = RANDOM_SEED):
    """Baseline for interpreting the SHAP-ranked curves above: average
    deletion/insertion curves over several *random* feature orderings.
    A faithful SHAP ranking should show a deletion curve that drops much
    faster than this random baseline, and an insertion curve that rises
    much faster — the raw AUC numbers alone aren't interpretable without
    this comparison."""
    rng = np.random.default_rng(seed)
    del_curves, ins_curves = [], []
    for _ in range(n_repeats):
        order = rng.permutation(n_features).tolist()
        del_curves.append(deletion_curve(model, X_sample, order, baseline))
        ins_curves.append(insertion_curve(model, X_sample, order, baseline))
    return np.mean(del_curves, axis=0), np.mean(ins_curves, axis=0)


def curve_auc(curve: np.ndarray) -> float:
    """Normalized area under the curve (trapezoidal), for a single
    summary number per curve — lower deletion AUC / higher insertion
    AUC both indicate a more faithful explanation."""
    x = np.linspace(0, 1, len(curve))
    trapz_fn = getattr(np, "trapezoid", np.trapz)  # numpy>=2.0 renamed trapz
    return float(trapz_fn(curve, x))


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    preprocessor = joblib.load(RESULTS / "preprocessor.joblib")
    model = joblib.load(RESULTS / "blackbox_model.joblib")
    feature_names = get_feature_names(preprocessor)

    X_train = preprocessor.transform(train_df[BASELINE_FEATURES])
    X_test = preprocessor.transform(test_df[BASELINE_FEATURES])
    baseline = get_baseline_values(X_train)

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(X_test.shape[0], size=min(SAMPLE_SIZE, X_test.shape[0]), replace=False)
    X_sample = X_test[sample_idx]

    shap_importance = pd.read_csv(RESULTS / "shap_importance.csv")
    # Map ranked feature names -> column indices in the processed matrix.
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    feature_order = [name_to_idx[f] for f in shap_importance["feature"] if f in name_to_idx]

    print(f"Running deletion/insertion over {len(feature_order)} features, "
          f"{len(sample_idx)} sampled test rows...")

    del_curve = deletion_curve(model, X_sample, feature_order, baseline)
    ins_curve = insertion_curve(model, X_sample, feature_order, baseline)
    del_auc = curve_auc(del_curve)
    ins_auc = curve_auc(ins_curve)

    print("Computing random-order baseline curves (10 random permutations)...")
    rand_del_curve, rand_ins_curve = random_order_curves(
        model, X_sample, len(feature_order), baseline, n_repeats=10
    )
    rand_del_auc = curve_auc(rand_del_curve)
    rand_ins_auc = curve_auc(rand_ins_curve)

    print(f"\nSHAP-order   deletion AUC: {del_auc:.4f}   insertion AUC: {ins_auc:.4f}")
    print(f"Random-order deletion AUC: {rand_del_auc:.4f}   insertion AUC: {rand_ins_auc:.4f}")
    print(f"\nFaithfulness gap — deletion (want SHAP << random): {del_auc - rand_del_auc:+.4f}")
    print(f"Faithfulness gap — insertion (want SHAP >> random): {ins_auc - rand_ins_auc:+.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "deletion_curve.npy", del_curve)
    np.save(RESULTS / "insertion_curve.npy", ins_curve)
    np.save(RESULTS / "deletion_curve_random.npy", rand_del_curve)
    np.save(RESULTS / "insertion_curve_random.npy", rand_ins_curve)
    pd.DataFrame([{
        "deletion_auc_shap": del_auc, "insertion_auc_shap": ins_auc,
        "deletion_auc_random": rand_del_auc, "insertion_auc_random": rand_ins_auc,
    }]).to_csv(RESULTS / "faithfulness_summary.csv", index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps = np.arange(len(del_curve))

    axes[0].plot(steps, del_curve, label=f"SHAP order (AUC={del_auc:.3f})", color="crimson")
    axes[0].plot(steps, rand_del_curve, label=f"Random order (AUC={rand_del_auc:.3f})",
                 color="gray", linestyle="--")
    axes[0].set_xlabel("Features masked")
    axes[0].set_ylabel("Mean predicted probability")
    axes[0].set_title("Deletion curve")
    axes[0].legend()

    axes[1].plot(steps, ins_curve, label=f"SHAP order (AUC={ins_auc:.3f})", color="steelblue")
    axes[1].plot(steps, rand_ins_curve, label=f"Random order (AUC={rand_ins_auc:.3f})",
                 color="gray", linestyle="--")
    axes[1].set_xlabel("Features inserted")
    axes[1].set_ylabel("Mean predicted probability")
    axes[1].set_title("Insertion curve")
    axes[1].legend()

    fig.suptitle("Faithfulness: SHAP ranking vs. random-order baseline")
    fig.tight_layout()
    fig.savefig(FIGURES / "faithfulness_curves_draft.png", dpi=150)
    print(f"\nSaved curves, summary, and a draft plot to {RESULTS} / {FIGURES}")


if __name__ == "__main__":
    main()
