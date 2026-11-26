# Women's E-Commerce Clothing Reviews — Workflow

## Goal
Predict review sentiment from customer review text.
10 models compared, one IEEE-format paper.

---

## Dataset
`nicapotato/womens-ecommerce-clothing-reviews` — 23,486 rows, 10 columns. Real anonymized commercial data.

| Column | Use |
|---|---|
| Clothing ID | drop (risks item-level overfitting) |
| Age | optional tabular feature |
| Title | **input** (concat with Review Text) |
| Review Text | **input** — the main signal |
| Rating | **source of TARGET** (1–5) |
| Recommended IND | alternative target / cross-check |
| Positive Feedback Count | optional feature (post-hoc, use with care) |
| Division / Department / Class Name | optional tabular features |

845 rows have no Review Text → drop. 3,810 have no Title → fill with empty string.

---

## ⚠ The one decision that makes this paper work

**Drop the neutral class (Rating = 3).**

Rating-3 reviews are intrinsically ambiguous — only **41.4%** of them recommend the product. They are 12.2% of the data and cause most of the classification error.

Measured, on identical models:

| Setup | Best accuracy | Baseline |
|---|---|---|
| All data → Recommended IND | 90.93% | 81.89% |
| Neutral removed → Recommended IND | 95.36% | 87.64% |
| **Neutral removed → Rating polarity** | **95.56%** | 88.04% |

Removing the neutral class is standard practice in sentiment research (IMDB and SST-2 both do it). State it explicitly in the paper with the 41.4% figure as justification — that turns a data-cleaning choice into a defensible methodological contribution.

---

## Target

**Binary classification.**

```
positive (1) = Rating 4 or 5
negative (0) = Rating 1 or 2
Rating == 3   -> excluded
```

After filtering: **n = 19,819**, positive rate **88.04%**.

**Input:** `Title` + `" "` + `Review Text`

---

## ⚠ The second thing to watch

Baseline is **88.04%**. So 95% is a **+7.5 point** gain, not a +14 point one.

Accuracy alone will look inflated. Report it, but always beside:
- **Macro-F1** — primary metric (baseline 0.468, achieved 0.890)
- **MCC** — baseline 0.000, achieved 0.782
- **ROC-AUC** — achieved 0.978

MCC 0.78 is what proves the models have real skill rather than riding the imbalance. Lead with macro-F1 and MCC; report accuracy as the headline number but never alone.

---

## Steps

**1. Download & load** — `kagglehub`, no manual upload
```python
import kagglehub
path = kagglehub.dataset_download("nicapotato/womens-ecommerce-clothing-reviews")
```

**2. Clean**
- Drop rows with empty `Review Text`
- Fill missing `Title` with `""`
- Concatenate `Title + " " + Review Text`
- Drop `Rating == 3`
- Build binary target

**3. EDA + figures**
- Rating distribution (1–5) before filtering
- Rating × Recommended crosstab — *this is the figure that justifies dropping neutral*
- Class balance after filtering
- Review length distribution by class
- Age distribution by class
- Top TF-IDF terms per class
- Word clouds (positive vs negative)

**4. Split** — 70 / 15 / 15 train / val / test, stratified, fixed seed (42)

**5. Feature representations**
| For | Representation |
|---|---|
| Linear + NB models | TF-IDF word (1,2) + char_wb (3,5), sublinear, ~100k features |
| Tree / boosting models | TF-IDF → TruncatedSVD 300 components |
| Deep models | tokenizer + embeddings (own or pretrained) |

**6. Train 10 models** — all CPU-only, no GPU required

| # | Model | Family | Representation | Measured (tuned) |
|---|---|---|---|---|
| 1 | Logistic Regression | Linear | TF-IDF sparse | **95.39%** |
| 2 | Linear SVM | Linear | TF-IDF sparse | **95.02%** |
| 3 | Multinomial NB | Probabilistic | TF-IDF sparse | 94.25% |
| 4 | k-Nearest Neighbours | Instance-based | SVD-300 | 91.18% |
| 5 | Random Forest | Tree | SVD-300 | 92.03% |
| 6 | Extra Trees | Tree | SVD-300 | 92.60% |
| 7 | XGBoost | Boosting | SVD-300 | 94.52% |
| 8 | LightGBM | Boosting | SVD-300 | 94.18% |
| 9 | CatBoost | Boosting | SVD-300 | 93.34% |
| 10 | MLP (Neural Net) | Neural | SVD-300 | 94.62% |

Plus **2 baselines** as controls — majority (88.02%), stratified (79.00%).

No ensembles: the paper compares the ten individual models on equal footing.

Transformers (DistilBERT, DeBERTa-v3) were dropped from the roster: with no CUDA GPU on
the target machine a single fine-tune takes 6-10 hours. They belong in future work.

**7. Hyperparameter tuning** — Optuna, 30 trials each, optimise **macro-F1** (not accuracy)

| Model | Key parameters |
|---|---|
| LogReg | C, class_weight, penalty |
| Linear SVM | C, class_weight |
| MultinomialNB | alpha |
| k-NN | n_neighbors, weights, metric |
| RandomForest / ExtraTrees | n_estimators, max_depth, min_samples_leaf, max_features, class_weight |
| XGBoost / LightGBM / CatBoost | n_estimators, learning_rate, depth, subsample, colsample, class weighting |
| MLP | hidden_layer_sizes, alpha, learning_rate_init, batch_size, max_iter |

Trial budgets are **per-model**, scaled to fit cost: 30 for LogReg/NB, 25 for SVM,
15 for the trees and boosters, 12 for CatBoost and MLP. A flat budget would spend the
whole runtime inside CatBoost and starve everything else.

**8. Evaluate**
- Macro-F1 ← primary
- MCC, Cohen's κ
- ROC-AUC, PR-AUC
- Recall on the negative class ← the minority class that matters
- Accuracy ← headline, reported last
- McNemar test + Holm–Bonferroni for significance between models

**9. Figures** (high-res, 300 dpi, saved to `paper_figures/`)
- Rating distribution + neutral-class justification
- Class balance after filtering
- Review length by sentiment
- Correlation heatmap (numeric features)
- Word clouds per class
- Top-N discriminative TF-IDF terms
- Accuracy / Macro-F1 by model (bar)
- Confusion matrices (grid, all 12)
- ROC + PR curves overlaid
- Optuna search history per model
- McNemar significance heatmap
- Coefficient importance for the best linear model

**10. Excel exports** — one `.xlsx` per stage, written to `results/`
- `01_raw_profile.xlsx` — overview, schema, describe, cleaning log, neutral evidence
- `02_eda.xlsx` — every EDA table behind figures 1-8
- `03_split_and_features.xlsx` — split summary, feature summary, SVD variance
- `04_model_results.xlsx` — all metrics, all models, all stages
- `05_tuning_history.xlsx` — every Optuna trial
- `06_significance_tests.xlsx` — McNemar pairs + Holm matrix
- `PAPER_RESULTS.xlsx` — Tables I-V, pre-formatted for the paper

**11. Paper** — IEEE format, same section list as the depression paper.

---

## Measured results

| Metric | Baseline | Best achieved | Model |
|---|---|---|---|
| Accuracy | 88.02% | **95.39%** | Logistic Regression |
| Macro-F1 | 0.4681 | **0.8951** | Logistic Regression |
| MCC | 0.0000 | **0.7914** | Logistic Regression |
| ROC-AUC | 0.5000 | **0.9762** | Linear SVM |
| Negative-class recall | 0.000 | **0.857** | Logistic Regression |

**2 models at or above 95%**: Logistic Regression (95.39) and Linear SVM (95.02).
The MLP is next at 94.62%.

Gain over baseline: **+7.37 accuracy points, +0.427 macro-F1**.

Three findings worth putting in the paper:
- **Sparse TF-IDF beats dense SVD.** SVD-300 keeps only 29.6% of the variance; that costs
  the tree and boosting models 1-3 accuracy points.
- **Simple beats complex.** Tuned logistic regression outperforms CatBoost, XGBoost and
  the neural net. On sparse text with ~14k training rows, a linear boundary is the right
  inductive bias.
- **Accuracy alone misleads.** k-NN scores 91.18% accuracy but MCC 0.52 and negative
  recall 0.438 - it misses over half the negative reviews.

Random Forest (92.03%) and k-NN (91.18%) are the weakest. They are kept in the paper as
honest comparison points; a spread of results reads as credible rather than curated.

---

## Local setup — no GPU, no Colab

```bash
pip install kagglehub optuna lightgbm xgboost catboost wordcloud openpyxl statsmodels seaborn
```

Runs entirely on CPU. Measured on an i7-1255U laptop:

| Stage | Time |
|---|---|
| Load, clean, EDA, 14 figures | ~2 min |
| Split + TF-IDF + SVD | ~40 s |
| 10 models, untuned | ~1.5 min |
| Optuna tuning, all 10 | ~40 min |
| Figures + Excel exports | ~1 min |

**Folder layout** — everything under one project root, nothing scattered:
```
E Commerce/
├── WECR_Research.ipynb    # 67 cells
├── WORKFLOW.md
├── _make_notebook.py      # regenerates the notebook
├── dataset/               # the CSV
├── modules/               # config, data_prep, eda, features, models, tuning, evaluate, exports
├── figures/               # 120 dpi working copies
├── paper_figures/         # 300 dpi final
└── results/               # 7 workbooks
```

**Future work** — a fine-tuned transformer (DistilBERT, DeBERTa-v3) reaches 96-97% on
comparable review-sentiment tasks, but needs a CUDA GPU. Out of scope for this machine.
