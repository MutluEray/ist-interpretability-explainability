# Interpretability vs. Explainability — Stroke Outcome Prediction (IST)

Does a post-hoc explanation of a black-box model actually tell you the same
thing an inherently interpretable model's own logic would? This project
tests that question directly on a real clinical task, rather than assuming
the answer — comparing **inherently interpretable models** (logistic
regression, a shallow decision tree) against **post-hoc explainability
methods** (SHAP, permutation importance) applied to a black-box
gradient-boosted model, all trained on identical baseline clinical features
from the International Stroke Trial (IST) database to predict 6-month
death/dependency after acute stroke.

Most clinical machine learning papers that use SHAP stop at "here is what
SHAP says matters about our black-box model." This project asks four
further questions: do interpretable and post-hoc explanations actually
*agree*? Is the post-hoc explanation *faithful* to what the black-box model
is actually doing? Is it *stable* across different training samples? And
*why* do the paradigms diverge where they do?

## Background

- **Interpretability**: a model whose reasoning can be read directly — a
  logistic regression coefficient, a decision tree split. There is nothing
  to explain after the fact; the model *is* the explanation.
- **Explainability**: a black-box model paired with a separate, post-hoc
  method (SHAP, permutation importance) that estimates what the black box
  is doing, without direct access to its internal logic.

A large literature (see [References](#references)) treats these as
interchangeable routes to the same kind of trust. This project measures
whether that's actually true, in one clinical setting.

## Data

Source: International Stroke Trial database (version 2), University of
Edinburgh, Edinburgh DataShare. DOI: https://doi.org/10.7488/ds/104
License: Open Data Commons Attribution License (ODC-By) v1.0. 19,435
patients, open access, no registration required.

Put the downloaded CSV in `data/raw/` (e.g. `data/raw/IST_corrected.csv`).
Nothing in `data/` is version-controlled — see `.gitignore`.

## Target and features

Primary target: `OCCODE` (6-month outcome), dichotomized as
death/dependency (`OCCODE` in {1,2}) vs. not (`OCCODE` in {3,4}), with
missing codes (0, 9) dropped. Class balance: 62.9% death/dependency, 37.1%
not — reasonably balanced, so AUROC is used as the primary metric.

Features: baseline/randomisation variables only (`RDELAY`, `RCONSC`, `SEX`,
`AGE`, `RSLEEP`, `RATRIAL`, `RCT`, `RVISINF`, `RHEP24`, `RASP3`, `RSBP`,
`RDEF1`–`RDEF8`, `STYPE`) — everything known within 48 hours of stroke
onset, before treatment is assigned.

Excluded by design:
- `RXASP`, `RXHEP` — randomized treatment allocation, not baseline patient
  state.
- `EXPDD`, `EXPD6`, `EXPD14` — pre-computed prognostic-score probabilities;
  including these would be leakage (grading another model, not learning
  from raw clinical variables).
- `COUNTRY` — proxy for site/healthcare-system variation, not patient-level
  clinical risk. Empirically confirmed as the right call: several countries
  have under 100 (down to single-digit) patients, which produced unstable,
  separation-driven logistic regression coefficients dominating the top of
  the importance table for reasons unrelated to clinical signal. Removing
  it cost a small amount of AUROC (0.823→0.804 for logistic regression) —
  accepted deliberately, in exchange for a feature set that's actually
  clinically defensible.
- All post-randomisation / 14-day discharge-form fields.

## Methods

Two models, trained on identical preprocessed features (standardized
numeric, one-hot encoded categorical):

- **Interpretable**: logistic regression + a depth-4 decision tree, native
  feature importances read directly from coefficients / split structure.
- **Black-box**: `HistGradientBoostingClassifier` (scikit-learn), explained
  post-hoc with **TreeSHAP** and **permutation importance**.

Four analyses, run in sequence:

1. **Agreement** — Spearman rank correlation between all four importance
   rankings (LR, tree, SHAP, permutation).
2. **Faithfulness** — deletion/insertion curves (Petsiuk et al., 2018):
   mask/reveal features in SHAP-ranked order and track the black-box
   model's predicted probability, compared against a random-order baseline.
3. **Stability** — the black-box model is refit on 50 bootstrap resamples
   of the training data (fixed hyperparameters, fixed evaluation sample),
   TreeSHAP recomputed each time; stability is the mean pairwise Spearman
   correlation across the 50 resulting rankings.
4. **Mechanism** — inspecting *why* the interpretable models' native
   importances diverge from the black box's post-hoc explanation.

## Results

| Model | AUROC | AUPRC |
|---|---|---|
| Logistic regression | 0.804 | 0.873 |
| Decision tree (depth 4) | 0.762 | 0.824 |
| HistGradientBoosting (black-box) | 0.806 | 0.875 |

**The black-box model barely outperforms plain logistic regression.** With
a feature set of mostly binary clinical flags, there's little nonlinear
structure for a black-box model's extra capacity to exploit — the
relationship between these baseline variables and 6-month outcome is close
to additive.

**Agreement (Spearman rank correlation):**

| | LogReg | Tree | SHAP | Permutation |
|---|---|---|---|---|
| **LogReg** | 1.000 | 0.291 | 0.231 | 0.425 |
| **Tree** | 0.291 | 1.000 | 0.430 | 0.460 |
| **SHAP** | 0.231 | 0.430 | 1.000 | 0.752 |
| **Permutation** | 0.425 | 0.460 | 0.752 | 1.000 |

SHAP and permutation importance — two post-hoc methods explaining the
*same* black-box model — agree strongly (0.752). The interpretable models'
own importances agree with those post-hoc explanations only moderately
(0.231–0.460). Two models with near-identical accuracy weight the clinical
evidence differently, depending on which method you ask.

**Faithfulness (deletion/insertion vs. random-order baseline):**

| | Deletion AUC (lower = better) | Insertion AUC (higher = better) |
|---|---|---|
| SHAP order | 0.489 | 0.599 |
| Random order | 0.544 | 0.556 |

SHAP's ranking beats random ordering in the correct direction on both
tests, but modestly. Consistent with several of the deficit-flag variables
(`RDEF1`–`RDEF8`) carrying correlated, overlapping clinical signal — no
single feature is irreplaceable, so masking any one of them doesn't
collapse the prediction the way it would if it were the model's only
source of that information.

**Stability:** mean pairwise Spearman correlation across 50 bootstrap
resamples = **0.805** — the SHAP ranking is highly reproducible across
different training samples, not an artifact of which patients happened to
be in the training set.

**Mechanism:** the depth-4 decision tree structurally uses only ~5–6 of the
~20 clinical variables in its splits — it assigns exactly zero importance
to most features, not merely low importance. Logistic regression assigns
every feature *some* coefficient, but that coefficient reflects effect size
per unit, with no weighting for how often a feature actually applies — a
rare-but-severe predictor (e.g. `RCONSC_U`, "unconscious at randomisation,"
260/19,435 patients) can dominate a coefficient table while barely
registering in SHAP's prevalence-weighted average-impact view. Two distinct
structural reasons for the disagreement in the table above, not one.

See `figures/` for the corresponding plots: `importance_comparison.png`,
`faithfulness_curves.png`, `stability.png`.

## Discussion

Put together, this isn't a story of "SHAP is good" or "SHAP is bad." SHAP's
explanation of the black-box model here is internally consistent (agrees
with permutation importance), reproducible (0.805 stability), and faithful
above a random-order baseline — and it is *still* not interchangeable with
what a transparent model's own logic says. That distinction — a
well-behaved post-hoc explanation is not automatically equivalent to
interpretability — is the actual finding, and it's sharpened by the fact
that the black-box model barely outperforms logistic regression in the
first place: if a fully transparent model gets you nearly all the way
there, the bar for "we need the black box, and we trust our explanation of
it" should be high, and a stable/faithful/consistent explanation doesn't
automatically clear that bar.

This directly tests the argument in Rudin (2019): black-box models should
be avoided in high-stakes settings whenever an equally accurate
interpretable model exists — and empirically demonstrates the
["disagreement problem"](https://arxiv.org/abs/2202.01602) (Krishna et al.,
2022) between explanation methods on real clinical data, with a concrete,
traceable mechanism (tree sparsity + prevalence-unweighted coefficients)
rather than just an observed correlation number.

## Limitations

- **Faithfulness protocol.** The deletion/insertion curves mask features to
  a baseline value and re-predict *without retraining* the model. Hooker et
  al. (2019) showed this protocol is confounded by distribution shift: a
  prediction drop after masking may partly reflect the model behaving
  oddly on out-of-distribution inputs rather than purely reflecting feature
  importance. The more rigorous fix (ROAR — retrain the model after each
  masking step) was not used here for computational tractability within a
  one-week project scope; the faithfulness numbers above should be read
  with that caveat.
- **"Stability" here means training-sample stability** (refit on bootstrap
  resamples of the training data, fixed evaluation set) — a different,
  related concept from the *local input-perturbation robustness* studied by
  Alvarez-Melis & Jaakkola (2018), which asks whether similar individual
  patients get similar explanations from a fixed model. This project does
  not test that second notion of robustness.
- **Single dataset, single outcome.** Results are specific to this feature
  set, this outcome definition (6-month death/dependency), and this
  black-box model family (gradient-boosted trees). They should not be
  read as a general claim about SHAP or about gradient boosting.

## References

- Rudin, C. (2019). Stop explaining black box machine learning models for
  high stakes decisions and use interpretable models instead. *Nature
  Machine Intelligence*, 1(5), 206–215.
- Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S., &
  Lakkaraju, H. (2022). The disagreement problem in explainable machine
  learning: a practitioner's perspective. arXiv:2202.01602.
- Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized input
  sampling for explanation of black-box models. *BMVC*.
- Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark
  for interpretability methods in deep neural networks (ROAR). *NeurIPS*.
- Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the robustness of
  interpretability methods. *ICML Workshop on Human Interpretability in
  Machine Learning*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions (SHAP). *NeurIPS*.
- Sandercock, P., Niewada, M., & Członkowska, A. (2011). The International
  Stroke Trial database. *Trials*, 12, 101.
- Fang, G., Huang, Z., & Wang, Z. (2022). Predicting ischemic stroke
  outcome using deep learning approaches. *Frontiers in Genetics*, 12,
  827522. (Same dataset and 6-month outcome target; uses SHAP for
  post-hoc explanation only, without the agreement/faithfulness/stability
  comparison this project adds.)

## Repo layout

```
ist-interpretability-explainability/
├── data/
│   ├── raw/            # put the downloaded CSV here (gitignored)
│   └── processed/      # cleaned train/test splits land here (gitignored)
├── src/
│   ├── data_cleaning.py       # Day 1
│   ├── target_definition.py   # Day 1
│   ├── train_interpretable.py # Day 2 — logistic regression + shallow tree
│   ├── train_blackbox.py      # Day 3 — HistGradientBoostingClassifier (sklearn)
│   ├── explainability.py      # Day 4 — SHAP (TreeSHAP) + permutation importance
│   ├── faithfulness.py        # Day 5 — deletion/insertion curves
│   ├── stability.py           # Day 6 — bootstrap resampling stability
│   ├── make_figures.py        # Day 6/7 — final figure generation
│   └── utils.py               # shared helpers (paths, seed, I/O)
├── figures/             # output figures land here (gitignored except .gitkeep)
├── results/             # metrics, importance tables, etc. (gitignored except .gitkeep)
├── requirements.txt
└── README.md
```

## Running order

```
python src/data_cleaning.py
python src/target_definition.py
python src/train_interpretable.py
python src/train_blackbox.py
python src/explainability.py
python src/faithfulness.py
python src/stability.py
python src/make_figures.py
```

Each script reads from `data/processed/` and/or `results/` and writes back
into `results/` and `figures/` — run them in order the first time through.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
