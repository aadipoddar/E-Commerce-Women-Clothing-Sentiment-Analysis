"""Result figures and pairwise significance testing."""
from __future__ import annotations

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve,
                             roc_curve, confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar

from config import CLASS_NAMES, DPI, PALETTE, figure
from models import matrices, metric_block, scores

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.autolayout": True, "savefig.bbox": "tight"})
NEG, POS = PALETTE["negative"], PALETTE["positive"]


def _save(fig, name: str) -> None:
    fig.savefig(figure(name, paper=True), dpi=DPI)
    fig.savefig(figure(name, paper=False), dpi=120)
    plt.close(fig)
    print(f"   saved {name}")


# ------------------------------------------------------------------ figure 9
def model_comparison(lb: pd.DataFrame, baseline_acc: float,
                     baseline_f1: float) -> pd.DataFrame:
    d = lb.sort_values("test_macro_f1")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    colors = ["#7f8c8d" if "Baseline" in m else
              (POS if f >= 0.95 else "#2980b9")
              for m, f in zip(d["model"], d["test_accuracy"])]
    axes[0].barh(d["model"], d["test_accuracy"] * 100, color=colors,
                 edgecolor="black", lw=.6)
    axes[0].axvline(baseline_acc * 100, ls="--", lw=2, color=NEG,
                    label=f"majority baseline {baseline_acc*100:.2f}%")
    axes[0].axvline(95, ls=":", lw=2, color="black", label="95% target")
    for i, v in enumerate(d["test_accuracy"]):
        axes[0].text(v * 100 + .15, i, f"{v*100:.2f}", va="center", fontsize=11)
    axes[0].set_xlim(75, 100); axes[0].set_xlabel("Test accuracy (%)")
    axes[0].set_title("Accuracy by model"); axes[0].legend(fontsize=10, loc="lower right")

    axes[1].barh(d["model"], d["test_macro_f1"],
                 color=["#7f8c8d" if "Baseline" in m else "#8e44ad" for m in d["model"]],
                 edgecolor="black", lw=.6)
    axes[1].axvline(baseline_f1, ls="--", lw=2, color=NEG,
                    label=f"baseline macro-F1 {baseline_f1:.3f}")
    for i, v in enumerate(d["test_macro_f1"]):
        axes[1].text(v + .006, i, f"{v:.3f}", va="center", fontsize=11)
    axes[1].set_xlim(0.4, 1.0); axes[1].set_xlabel("Test macro-F1")
    axes[1].set_title("Macro-F1 by model  (primary metric)")
    axes[1].legend(fontsize=10, loc="lower right")
    _save(fig, "fig09_model_comparison.png")
    return d


# ------------------------------------------------------------------ figure 10
def confusion_grid(preds: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    names = list(preds)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_true, preds[name], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        acc = (preds[name] == y_true).mean()
        ax.set_title(f"{name}\nacc={acc*100:.2f}%", fontsize=12)
        ax.set_xlabel(""); ax.set_ylabel("")
        tn, fp, fn, tp = cm.ravel()
        rows.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (test set)", fontsize=17, y=1.0)
    _save(fig, "fig10_confusion_matrices.png")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure 11
def curves(probas: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import auc, average_precision_score, roc_auc_score
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
    rows = []
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(probas), 3)))
    for (name, p), c in zip(probas.items(), cmap):
        fpr, tpr, _ = roc_curve(y_true, p)
        ra = roc_auc_score(y_true, p)
        axes[0].plot(fpr, tpr, lw=2, color=c, label=f"{name} ({ra:.3f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        pa = average_precision_score(y_true, p)
        axes[1].plot(rc, pr, lw=2, color=c, label=f"{name} ({pa:.3f})")
        rows.append({"model": name, "roc_auc": round(ra, 4), "pr_auc": round(pa, 4)})
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1.5)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves"); axes[0].legend(fontsize=9, loc="lower right")
    axes[1].axhline(y_true.mean(), ls="--", color="grey", lw=1.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curves"); axes[1].legend(fontsize=9, loc="lower left")
    _save(fig, "fig11_roc_pr_curves.png")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure 12
def tuning_curves(hist: pd.DataFrame) -> None:
    models = hist["model"].unique()
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, models):
        d = hist[hist["model"] == m]
        ax.plot(d["number"], d["value"], "o", ms=4, alpha=.45, color="#2980b9")
        ax.plot(d["number"], d["running_best"], lw=2.2, color=NEG, label="best so far")
        ax.set_title(m, fontsize=12); ax.set_xlabel("trial"); ax.set_ylabel("val macro-F1")
        ax.legend(fontsize=9)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle("Optuna search history", fontsize=17, y=1.0)
    _save(fig, "fig12_tuning_history.png")


# ------------------------------------------------------------------ figure 13
def significance(preds: dict[str, np.ndarray], y_true: np.ndarray
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar for every pair, then Holm-Bonferroni over the whole family."""
    names = list(preds)
    rows = []
    for a, b in itertools.combinations(names, 2):
        ca, cb = preds[a] == y_true, preds[b] == y_true
        n01 = int((~ca & cb).sum())
        n10 = int((ca & ~cb).sum())
        table = [[int((ca & cb).sum()), n10], [n01, int((~ca & ~cb).sum())]]
        exact = (n01 + n10) < 25
        res = mcnemar(table, exact=exact, correction=not exact)
        rows.append({"model_a": a, "model_b": b, "a_only_correct": n10,
                     "b_only_correct": n01, "statistic": float(res.statistic),
                     "p_raw": float(res.pvalue)})
    d = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)

    m = len(d)
    adj, running = [], 0.0
    for i, p in enumerate(d["p_raw"]):
        running = max(running, min((m - i) * p, 1.0))
        adj.append(running)
    d["p_holm"] = adj
    d["significant_5pct"] = d["p_holm"] < 0.05

    arr = np.full((len(names), len(names)), np.nan, dtype=float)
    idx = {n: i for i, n in enumerate(names)}
    for _, r in d.iterrows():
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        arr[i, j] = arr[j, i] = r["p_holm"]
    np.fill_diagonal(arr, 1.0)
    matrix = pd.DataFrame(arr, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(1.15 * len(names) + 4, 1.0 * len(names) + 3))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=.2, square=True, linewidths=.6, ax=ax,
                cbar_kws={"label": "Holm-adjusted p"})
    ax.set_title("McNemar pairwise significance\n(green = models genuinely differ)")
    _save(fig, "fig13_significance.png")
    return d.round(6), matrix.round(4).reset_index(names="model")


# ------------------------------------------------------------------ figure 14
def top_coefficients(model, vectorizer, top_n: int = 20) -> pd.DataFrame:
    """Interpretability for the best sparse linear model."""
    est = getattr(model, "estimator", model)
    if hasattr(model, "calibrated_classifiers_"):
        est = model.calibrated_classifiers_[0].estimator
    coef = np.ravel(getattr(est, "coef_", None))
    if coef is None:
        return pd.DataFrame()
    vocab = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(coef)
    tbl = pd.concat([
        pd.DataFrame({"term": vocab[order[:top_n]], "coefficient": coef[order[:top_n]],
                      "direction": "negative"}),
        pd.DataFrame({"term": vocab[order[-top_n:]], "coefficient": coef[order[-top_n:]],
                      "direction": "positive"}),
    ]).round(4).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    s = tbl.sort_values("coefficient")
    ax.barh(s["term"], s["coefficient"],
            color=[NEG if d == "negative" else POS for d in s["direction"]],
            edgecolor="black", lw=.5)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Model coefficient  (negative <- -> positive)")
    ax.set_title(f"Top {top_n} terms per class, best linear model")
    _save(fig, "fig14_top_coefficients.png")
    return tbl
