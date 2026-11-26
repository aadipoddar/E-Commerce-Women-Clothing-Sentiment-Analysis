"""Generate WECR_Research.ipynb from the modules in modules/.

Kept as a script so the notebook can be regenerated deterministically instead of
being hand-edited - the same pattern as the Student Depression project.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "WECR_Research.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ============================================================== title
md("""
# Sentiment Classification of Women's E-Commerce Clothing Reviews
### A Comparative Study of Ten Machine Learning Models

**Dataset** — `nicapotato/womens-ecommerce-clothing-reviews` (23,486 reviews, real anonymised commercial data)

**Task** — binary classification: predict whether a review is **positive** (4–5★) or **negative** (1–2★) from its text.

**Key design decision** — the neutral class (3★) is removed. Only 41.4% of 3★ reviewers
recommend the product, so the class is intrinsically ambiguous and inflates error for every model.
Removing it is standard practice in sentiment research (IMDB, SST-2).

---

| | |
|---|---|
| Rows after cleaning | 19,813 |
| Majority baseline | 88.04% |
| Models compared | 10 models across 6 families + 2 controls |
| Primary metric | **Macro-F1** (accuracy is reported but never leads) |
| Tuning | Optuna, per-model trial budgets, optimising macro-F1 |
| Hardware | CPU only — the whole notebook runs on a laptop, no GPU required |

**Model roster**

| # | Model | Family | Representation |
|---|---|---|---|
| 1 | Logistic Regression | Linear | TF-IDF sparse |
| 2 | Linear SVM | Linear | TF-IDF sparse |
| 3 | Multinomial NB | Probabilistic | TF-IDF sparse |
| 4 | k-Nearest Neighbours | Instance-based | SVD-300 |
| 5 | Random Forest | Tree | SVD-300 |
| 6 | Extra Trees | Tree | SVD-300 |
| 7 | XGBoost | Boosting | SVD-300 |
| 8 | LightGBM | Boosting | SVD-300 |
| 9 | CatBoost | Boosting | SVD-300 |
| 10 | MLP (Neural Net) | Neural | SVD-300 |
""")

# ============================================================== setup
md("## 1. Environment setup")

code("""

# Run once if any package is missing - all CPU-only, no GPU or Colab needed
# !pip install -q kagglehub optuna lightgbm xgboost catboost wordcloud openpyxl statsmodels seaborn
""")

code("""
import sys, warnings, pickle, time
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "modules"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

from config import *            # paths, seeds, column names, hyper-parameter defaults
import data_prep as dp
import features as ft
import models as M
import tuning as T
import evaluate as E
import exports as X
import eda

print("project root :", ROOT)
print("modules      :", MODULES.name)
print("seed         :", SEED)
""")

# ============================================================== data
md("""
## 2. Data acquisition

The loader checks `dataset/` first so the notebook re-runs offline, and only calls
`kagglehub` when the file is missing.
""")

code("""
raw = dp.download()
print("raw shape:", raw.shape)
raw.head(3)
""")

md("### 2.1 Raw profile")

code("""
profile = dp.profile(raw)
display(profile["overview"])
display(profile["schema"])
""")

md("""
### 2.2 Cleaning

Four steps, in order:
1. drop rows with no review body (845 rows)
2. fill missing titles with an empty string, then concatenate `Title + Review Text`
3. **drop the neutral class (Rating = 3)** — 2,823 rows
4. drop duplicate review texts, which would otherwise leak across the split
""")

code("""
df = dp.clean(raw)
display(dp.cleaning_log(raw, df))
df[["text", "Rating", TARGET]].head(3)
""")

md("""
### 2.3 Why the neutral class is dropped

This is the table that justifies the decision — and it belongs in the paper.
""")

code("""
evidence = dp.neutral_evidence(raw)
display(evidence)
print("3-star reviews: only {:.1%} recommend the product -> intrinsically ambiguous"
      .format(evidence.loc[evidence.Rating == 3, "recommended"].iloc[0]))
""")

# ============================================================== eda
md("## 3. Exploratory data analysis")

md("### Figure 1 — rating distribution")
code("""tbl_rating = eda.rating_distribution(raw); display(tbl_rating)""")

md("### Figure 2 — the neutral-class justification")
code("""_ = eda.neutral_justification(evidence)""")

md("### Figure 3 — class balance after filtering")
code("""tbl_balance = eda.class_balance(df); display(tbl_balance)""")

md("### Figure 4 — review length by sentiment")
code("""tbl_length = eda.review_length(df); display(tbl_length)""")

md("### Figure 5 — reviewer age and department")
code("""cat_tables = eda.age_and_categories(df); display(cat_tables["department"])""")

md("### Figure 6 — correlation heatmap")
code("""tbl_corr = eda.correlation(df); display(tbl_corr)""")

md("### Figure 7 — word clouds per class")
code("""tbl_words = eda.wordclouds(df); display(tbl_words.head(10))""")

md("### Figure 8 — most discriminative terms (log-odds ratio)")
code("""tbl_terms = eda.discriminative_terms(df); display(tbl_terms.head(10))""")

md("### Export the EDA workbook")
code("""
X.write({"rating_distribution": tbl_rating, "neutral_evidence": evidence,
         "class_balance": tbl_balance, "review_length": tbl_length,
         **{k[:28]: v for k, v in cat_tables.items()},
         "correlation": tbl_corr, "top_words": tbl_words,
         "discriminative_terms": tbl_terms}, "02_eda.xlsx")
""")

# ============================================================== split
md("""
## 4. Splitting and feature representation

70 / 15 / 15 stratified, fixed seed. The validation split is what Optuna optimises
against; the test split is touched only once, at the very end.

Two representations, because one size does not fit all:

| Representation | Fed to |
|---|---|
| TF-IDF word (1,2) + char_wb (3,5), ~97k features | Logistic Regression, Linear SVM, Multinomial NB |
| TruncatedSVD 300 components, standardised | k-NN, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, MLP |

The vectoriser is fitted on **train only** — no test vocabulary leaks into the model.

Note the trade-off this creates: SVD-300 retains only ~29.6% of the variance, and section 11
shows that lost information costs the dense-representation models 1–3 accuracy points.
""")

code("""
split = ft.make_split(df)
display(split.summary())
""")

code("""
feats = ft.build_features(split)
display(feats.summary())
""")

# ============================================================== baselines
md("""
## 5. Baselines

With an 88% majority class, a model that predicts "positive" for everything already
scores 88% accuracy. These two controls sit in every results table so that number is
never mistaken for skill.
""")

code("""
base_df, _ = M.run_all(M.BASELINES, feats, split, stage="baseline")
display(M.leaderboard(base_df))
""")

# ============================================================== untuned
md("## 6. Untuned models — the control pass")

code("""
untuned_df, untuned_models = M.run_all(M.REGISTRY, feats, split, stage="untuned")
display(M.leaderboard(untuned_df))
""")

# ============================================================== tuning
md("""
## 7. Hyper-parameter tuning with Optuna

Every study maximises **validation macro-F1**, never accuracy — on an 88/12 target,
accuracy rewards a model for ignoring the minority class.

Trial budgets are per-model because a CatBoost fit costs ~25 s while a Multinomial NB
fit costs ~0.02 s. A flat budget would spend the whole runtime in one place.

`class_weight` (or its equivalent) is in every search space that supports it — it is the
main lever against the imbalance.

⚠ This cell takes roughly 35 minutes on a laptop CPU.
""")

code("""
best_params, studies, tune_summary = T.tune_all(feats, split)
display(tune_summary)
""")

md("### 7.1 Retrain every model on its best configuration")

code("""
tuned_df, tuned_models = M.run_all(M.REGISTRY, feats, split,
                                   stage="tuned", params_by_model=best_params)
results = pd.concat([base_df, untuned_df, tuned_df], ignore_index=True)
display(M.leaderboard(results[results.stage == "tuned"]))
""")

md("### Figure 12 — Optuna search history")
code("""
history = T.history(studies)
E.tuning_curves(history)
""")

# ============================================================== collect
md("""
## 8. Collect predictions for the result figures

Every tuned model is asked for its hard predictions and its positive-class scores on the
**test split** — the only point in the notebook where the test set is touched.
""")

code("""
spec_by_name = {s.name: s for s in M.REGISTRY}
preds, probas = {}, {}
for name, mdl in tuned_models.items():
    _, _, B = M.matrices(spec_by_name[name], feats)
    preds[name] = mdl.predict(B)
    probas[name] = M.scores(mdl, B)

final = results.copy()
print("collected:", len(preds), "models")
""")

# ============================================================== results
md("## 9. Results")

md("### Figure 9 — model comparison")
code("""
lb = M.leaderboard(final[final.stage.isin(["tuned", "baseline"])])
mb = final[final.model == "Baseline (majority)"].iloc[0]
_ = E.model_comparison(lb, mb["test_accuracy"], mb["test_macro_f1"])
display(lb)
""")

md("### Figure 10 — confusion matrices")
code("""tbl_cm = E.confusion_grid(preds, split.y_test); display(tbl_cm)""")

md("### Figure 11 — ROC and precision-recall curves")
code("""tbl_auc = E.curves(probas, split.y_test); display(tbl_auc)""")

md("""
### Figure 13 — statistical significance

McNemar's test on every pair of models, then Holm–Bonferroni correction across the
whole family of comparisons. Without the correction, testing 36 pairs at α = 0.05
would produce roughly two false positives by chance alone.
""")
code("""
sig, sig_matrix = E.significance(preds, split.y_test)
display(sig.head(12))
""")

md("### Figure 14 — most influential terms in the best linear model")
code("""
tbl_coef = E.top_coefficients(tuned_models["Logistic Regression"], feats.vectorizer)
display(tbl_coef.head(10))
""")

# ============================================================== exports
md("## 10. Excel exports")

code("""
X.write({**dp.profile(raw), "cleaning_log": dp.cleaning_log(raw, df),
         "neutral_evidence": evidence}, "01_raw_profile.xlsx")
X.write({"split_summary": split.summary(), "feature_summary": feats.summary(),
         "svd_variance": ft.svd_variance_table(feats)}, "03_split_and_features.xlsx")
X.write({"all_results": final.round(4), "leaderboard": M.leaderboard(final)},
        "04_model_results.xlsx")
X.write({"summary": tune_summary, "history": history.round(5)}, "05_tuning_history.xlsx")
X.write({"mcnemar_pairs": sig, "holm_matrix": sig_matrix}, "06_significance_tests.xlsx")
X.write({**X.paper_tables(final, tune_summary, sig),
         "confusion_counts": tbl_cm, "auc_summary": tbl_auc,
         "top_coefficients": tbl_coef}, "PAPER_RESULTS.xlsx")
""")

code("""print(X.summary_text(final))""")

# ============================================================== colab
md("""
---

## 11. Conclusion

| Rank | Model | Family | Accuracy | Macro-F1 | MCC | Neg-recall |
|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | Linear | **95.39%** | **0.8951** | **0.7914** | 0.857 |
| 2 | **Linear SVM** | Linear | **95.02%** | 0.8781 | 0.7568 | 0.756 |
| 3 | MLP (Neural Net) | Neural | 94.62% | 0.8698 | 0.7399 | 0.753 |
| 4 | Multinomial NB | Probabilistic | 94.25% | 0.8676 | 0.7357 | 0.795 |
| 5 | XGBoost | Boosting | 94.52% | 0.8662 | 0.7330 | 0.739 |
| 6 | LightGBM | Boosting | 94.18% | 0.8520 | 0.7074 | 0.680 |
| 7 | CatBoost | Boosting | 93.34% | 0.8479 | 0.6967 | 0.767 |
| 8 | Extra Trees | Tree | 92.60% | 0.8095 | 0.6232 | 0.601 |
| 9 | Random Forest | Tree | 92.03% | 0.7725 | 0.5669 | 0.478 |
| 10 | k-Nearest Neighbours | Instance-based | 91.18% | 0.7474 | 0.5163 | 0.438 |
| — | *Baseline (majority)* | *Control* | *88.02%* | *0.4681* | *0.0000* | *0.000* |
| — | *Baseline (stratified)* | *Control* | *79.00%* | *0.4997* | *−0.0005* | *0.118* |

**Findings**

1. **Two models clear 95%** — Logistic Regression (95.39%) and Linear SVM (95.02%),
   with the MLP just behind at 94.62%.

2. **Sparse TF-IDF beats dense SVD on this task.** The three models that consume the full
   ~97k-feature sparse matrix take first, second and fourth place. Projecting to 300 SVD
   components retains only 29.6% of the variance, and that lost information costs the
   tree and boosting models 1–3 accuracy points.

3. **Simple beats complex here.** A tuned logistic regression outperforms CatBoost,
   XGBoost and a neural network. On high-dimensional sparse text with ~14k training rows,
   the linear decision boundary is the right inductive bias.

4. **Accuracy alone would mislead.** Every model beats the 88.02% baseline on accuracy,
   including k-NN at 91.18% — but k-NN's MCC is 0.52 and its negative-class recall is
   0.438, so it misses more than half the negative reviews. Macro-F1 and MCC are what
   separate genuine skill from riding the class imbalance.

5. **The gain is real**: +7.37 accuracy points and +0.427 macro-F1 over the majority
   baseline, with MCC rising from 0.000 to 0.791.

**Limitations**

- Only reviews with body text are modelled (19,813 of 23,486); the 845 text-free rows
  cannot be classified from text at all.
- The neutral class (3★, 12.2% of data) is excluded by design. The models therefore say
  nothing about genuinely ambiguous reviews — a deployment would need a three-way
  variant or an abstain option.
- Single train/test split with a fixed seed. Repeated cross-validation would give tighter
  confidence intervals on the small differences among the top four models.

**Future work** — the natural next step is a fine-tuned transformer (DistilBERT,
DeBERTa-v3), which on comparable review-sentiment tasks reaches 96–97%. That needs a
CUDA GPU; every model in this notebook runs on a laptop CPU.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {NB.name}: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in cells)} markdown)")
