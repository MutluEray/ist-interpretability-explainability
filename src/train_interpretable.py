"""
Day 2 — Train the inherently interpretable models: logistic regression
and a shallow decision tree, using the shared preprocessing pipeline
(utils.build_preprocessor) so they're comparable to the black-box
model trained in train_blackbox.py. Extract each model's native
feature importances (standardized LR coefficients; tree importances)
and save everything to RESULTS/.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import (
    TRAIN_CSV, TEST_CSV, BASELINE_FEATURES, TARGET_COL, RESULTS,
    RANDOM_SEED, build_preprocessor, get_feature_names,
)


def train_logreg(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_tree(X_train, y_train, max_depth=4):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name: str) -> dict:
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

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor)

    # --- Logistic regression ---
    logreg = train_logreg(X_train, y_train)
    logreg_metrics = evaluate(logreg, X_test, y_test, "LogisticRegression")
    logreg_importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": logreg.coef_[0],
        "abs_coefficient": abs(logreg.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)

    # --- Shallow decision tree ---
    tree = train_tree(X_train, y_train)
    tree_metrics = evaluate(tree, X_test, y_test, "DecisionTree(depth=4)")
    tree_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False)

    # --- Save everything ---
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, RESULTS / "preprocessor.joblib")
    joblib.dump(logreg, RESULTS / "logreg_model.joblib")
    joblib.dump(tree, RESULTS / "tree_model.joblib")
    logreg_importance.to_csv(RESULTS / "logreg_importance.csv", index=False)
    tree_importance.to_csv(RESULTS / "tree_importance.csv", index=False)
    pd.DataFrame([logreg_metrics, tree_metrics]).to_csv(
        RESULTS / "interpretable_metrics.csv", index=False
    )

    print("\nTop 10 logistic regression features (by |coefficient|):")
    print(logreg_importance.head(10).to_string(index=False))
    print("\nTop 10 decision tree features (by importance):")
    print(tree_importance.head(10).to_string(index=False))
    print(f"\nSaved models and importances to {RESULTS}")


if __name__ == "__main__":
    main()
