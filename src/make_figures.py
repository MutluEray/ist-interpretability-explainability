"""
Day 6/7 — Generate the final portfolio-quality figures from the saved
results in RESULTS/:

  1. importance_comparison.png — LR coefficients vs. tree importance
     vs. SHAP vs. permutation importance, normalized and grouped,
     for the top features by SHAP ranking.
  2. faithfulness_curves.png — deletion/insertion curves vs. the
     random-order baseline (from faithfulness.py).
  3. stability.png — bootstrap distribution of SHAP importance for
     the top features (from stability.py), annotated with the overall
     stability score.

Static figures only (see plan discussion re: skipping the interactive
demo) — clean, well-labeled static figures serve a PhD-committee skim
better than a demo widget, within a 1-week budget.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import RESULTS, FIGURES, get_feature_names

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def clean_name(feature: str) -> str:
    """cat__RCONSC_F -> RCONSC_F, num__AGE -> AGE — for readable axis labels."""
    return feature.replace("cat__", "").replace("num__", "")


def plot_importance_comparison(top_n: int = 12):
    logreg = pd.read_csv(RESULTS / "logreg_importance.csv").set_index("feature")["abs_coefficient"]
    tree = pd.read_csv(RESULTS / "tree_importance.csv").set_index("feature")["importance"]
    shap_imp = pd.read_csv(RESULTS / "shap_importance.csv").set_index("feature")["mean_abs_shap"]
    perm = pd.read_csv(RESULTS / "permutation_importance.csv").set_index("feature")["perm_importance_mean"]

    def normalize(s):
        return s / s.max()

    top_features = shap_imp.sort_values(ascending=False).head(top_n).index.tolist()

    df = pd.DataFrame({
        "LogReg (interpretable)": normalize(logreg),
        "Decision Tree (interpretable)": normalize(tree),
        "SHAP (post-hoc)": normalize(shap_imp),
        "Permutation (post-hoc)": normalize(perm),
    }).loc[top_features]
    df.index = [clean_name(f) for f in df.index]
    df = df.iloc[::-1]  # largest at top of horizontal chart

    fig, ax = plt.subplots(figsize=(9, 7))
    df.plot(kind="barh", ax=ax, width=0.8,
            color=["#4c72b0", "#55a868", "#c44e52", "#dd8452"])
    ax.set_xlabel("Normalized importance (relative to each method's own max)")
    ax.set_title(f"Feature importance across methods (top {top_n} by SHAP)")
    ax.legend(title=None, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "importance_comparison.png")
    plt.close(fig)
    print("Saved importance_comparison.png")


def plot_faithfulness_curves():
    del_curve = np.load(RESULTS / "deletion_curve.npy")
    ins_curve = np.load(RESULTS / "insertion_curve.npy")
    rand_del = np.load(RESULTS / "deletion_curve_random.npy")
    rand_ins = np.load(RESULTS / "insertion_curve_random.npy")
    summary = pd.read_csv(RESULTS / "faithfulness_summary.csv").iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps = np.arange(len(del_curve))

    axes[0].plot(steps, del_curve, color="#c44e52",
                 label=f"SHAP order (AUC={summary['deletion_auc_shap']:.3f})")
    axes[0].plot(steps, rand_del, color="gray", linestyle="--",
                 label=f"Random order (AUC={summary['deletion_auc_random']:.3f})")
    axes[0].set_xlabel("Features masked (ranked, most important first)")
    axes[0].set_ylabel("Mean predicted probability")
    axes[0].set_title("Deletion curve")
    axes[0].legend(frameon=False)

    axes[1].plot(steps, ins_curve, color="#4c72b0",
                 label=f"SHAP order (AUC={summary['insertion_auc_shap']:.3f})")
    axes[1].plot(steps, rand_ins, color="gray", linestyle="--",
                 label=f"Random order (AUC={summary['insertion_auc_random']:.3f})")
    axes[1].set_xlabel("Features inserted (ranked, most important first)")
    axes[1].set_ylabel("Mean predicted probability")
    axes[1].set_title("Insertion curve")
    axes[1].legend(frameon=False)

    fig.suptitle("Faithfulness: SHAP ranking vs. random-order baseline")
    fig.tight_layout()
    fig.savefig(FIGURES / "faithfulness_curves.png")
    plt.close(fig)
    print("Saved faithfulness_curves.png")


def plot_stability(top_n: int = 15):
    importances = np.load(RESULTS / "bootstrap_shap_importances.npy")
    stability_df = pd.read_csv(RESULTS / "stability_summary.csv")
    score = pd.read_csv(RESULTS / "stability_score.csv").iloc[0]["mean_pairwise_spearman"]

    preprocessor = joblib.load(RESULTS / "preprocessor.joblib")
    feature_names = get_feature_names(preprocessor)

    top = stability_df.head(top_n).iloc[::-1]  # reverse so most important lands at
                                                 # top of the chart, matching the
                                                 # convention in importance_comparison.png
    top_idx = [feature_names.index(f) for f in top["feature"]]
    labels = [clean_name(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [importances[:, i] for i in top_idx],
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="black"),
    )
    for box in bp["boxes"]:
        box.set(facecolor="#8ab3d1")
    ax.set_xlabel("Mean |SHAP value| across 50 bootstrap resamples")
    ax.set_title(f"Stability of top {top_n} features\n"
                 f"(mean pairwise Spearman ρ across resamples = {score:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "stability.png")
    plt.close(fig)
    print("Saved stability.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_importance_comparison()
    plot_faithfulness_curves()
    plot_stability()
    print(f"\nAll final figures written to {FIGURES}")


if __name__ == "__main__":
    main()
