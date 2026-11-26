"""Exploratory figures and their matching Excel tables.

Every function returns the dataframe it plotted, so the notebook can send the
same numbers to both a chart and a workbook sheet without recomputing them.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (COL_AGE, COL_RATING, COL_RECOMMEND, CAT_COLS, CLASS_NAMES,
                    DPI, PALETTE, TARGET, TEXT_COL, figure)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.autolayout": True, "savefig.bbox": "tight"})

NEG, POS = PALETTE["negative"], PALETTE["positive"]


def _save(fig, name: str, paper: bool = True) -> None:
    fig.savefig(figure(name, paper=paper), dpi=DPI)
    if paper:
        fig.savefig(figure(name, paper=False), dpi=120)
    plt.close(fig)
    print(f"   saved {name}")


# ------------------------------------------------------------------ figure 1
def rating_distribution(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw 1-5 rating counts, with the dropped neutral class highlighted."""
    vc = raw[COL_RATING].value_counts().sort_index()
    tbl = pd.DataFrame({"rating": vc.index, "count": vc.values,
                        "share": (vc.values / len(raw)).round(4)})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [NEG, NEG, "#7f8c8d", POS, POS]
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="black", lw=.6)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v + len(raw) * .008, f"{v:,}",
                ha="center", fontsize=12)
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Rating distribution  —  grey class (3-star) is excluded")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NEG, "#7f8c8d", POS)]
    ax.legend(handles, ["negative (1-2 star)", "excluded (3-star)", "positive (4-5 star)"],
              frameon=True, fontsize=11)
    _save(fig, "fig01_rating_distribution.png")
    return tbl


# ------------------------------------------------------------------ figure 2
def neutral_justification(evidence: pd.DataFrame) -> pd.DataFrame:
    """The figure that defends dropping Rating 3."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = evidence[COL_RATING].astype(str)
    ax.bar(x, evidence["recommended"] * 100, color=POS, label="recommends product")
    ax.bar(x, evidence["not_recommended"] * 100,
           bottom=evidence["recommended"] * 100, color=NEG, label="does not recommend")
    for i, r in evidence.iterrows():
        ax.text(i, r["recommended"] * 100 / 2, f"{r['recommended'] * 100:.1f}%",
                ha="center", color="white", fontsize=12, weight="bold")
    ax.axvline(2, ls="--", lw=2, color="black")
    ax.annotate("3-star is genuinely ambiguous\nonly 41.4% recommend",
                xy=(2, 50), xytext=(2.75, 62), fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Share of reviews (%)")
    ax.set_title("Why the neutral class is removed")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(0, 100)
    _save(fig, "fig02_neutral_justification.png")
    return evidence


# ------------------------------------------------------------------ figure 3
def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    vc = df[TARGET].value_counts().sort_index()
    tbl = pd.DataFrame({"class": CLASS_NAMES, "count": vc.values,
                        "share": (vc.values / len(df)).round(4)})

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(CLASS_NAMES, vc.values, color=[NEG, POS], edgecolor="black", lw=.6)
    for i, v in enumerate(vc.values):
        ax[0].text(i, v + len(df) * .01, f"{v:,}", ha="center", fontsize=12)
    ax[0].set_ylabel("Number of reviews")
    ax[0].set_title("Class balance after filtering")
    ax[1].pie(vc.values, labels=CLASS_NAMES, colors=[NEG, POS], autopct="%1.1f%%",
              startangle=90, wedgeprops=dict(edgecolor="white", lw=2),
              textprops=dict(fontsize=13))
    ax[1].set_title(f"Majority baseline = {vc.max() / vc.sum() * 100:.2f}%")
    _save(fig, "fig03_class_balance.png")
    return tbl


# ------------------------------------------------------------------ figure 4
def review_length(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(n_words=df[TEXT_COL].str.split().str.len(),
                  n_chars=df[TEXT_COL].str.len())
    tbl = (d.groupby(d[TARGET].map({0: "negative", 1: "positive"}))[["n_words", "n_chars"]]
             .agg(["mean", "median", "std", "max"]).round(2))
    tbl.columns = ["_".join(c) for c in tbl.columns]
    tbl = tbl.reset_index(names="class")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, c in ((0, NEG), (1, POS)):
        sns.kdeplot(d.loc[d[TARGET] == lbl, "n_words"], ax=ax[0], color=c,
                    fill=True, alpha=.35, lw=2, label=CLASS_NAMES[lbl])
    ax[0].set_xlabel("Words per review"); ax[0].set_ylabel("Density")
    ax[0].set_title("Review length by sentiment"); ax[0].legend()
    sns.boxplot(data=d, x=d[TARGET].map({0: "negative", 1: "positive"}),
                y="n_words", hue=d[TARGET].map({0: "negative", 1: "positive"}),
                palette=[NEG, POS], ax=ax[1], legend=False)
    ax[1].set_xlabel(""); ax[1].set_ylabel("Words per review")
    ax[1].set_title("Spread of review length")
    _save(fig, "fig04_review_length.png")
    return tbl


# ------------------------------------------------------------------ figure 5
def age_and_categories(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    for lbl, c in ((0, NEG), (1, POS)):
        sns.kdeplot(df.loc[df[TARGET] == lbl, COL_AGE], ax=axes[0], color=c,
                    fill=True, alpha=.35, lw=2, label=CLASS_NAMES[lbl])
    axes[0].set_title("Reviewer age by sentiment")
    axes[0].set_xlabel("Age"); axes[0].legend()
    out["age"] = (df.groupby(df[TARGET].map({0: "negative", 1: "positive"}))[COL_AGE]
                    .agg(["mean", "median", "std"]).round(2).reset_index(names="class"))

    dept = (df.groupby("Department Name")[TARGET].agg(["mean", "size"])
              .sort_values("size", ascending=False))
    dept.columns = ["positive_rate", "n"]
    axes[1].barh(dept.index, dept["positive_rate"] * 100, color=POS, edgecolor="black", lw=.6)
    axes[1].axvline(df[TARGET].mean() * 100, ls="--", color="black", lw=2,
                    label=f"overall {df[TARGET].mean() * 100:.1f}%")
    axes[1].set_xlabel("Positive rate (%)")
    axes[1].set_title("Positive rate by department"); axes[1].legend(fontsize=10)
    out["department"] = dept.round(4).reset_index()
    _save(fig, "fig05_age_and_department.png")

    for col in CAT_COLS:
        g = df.groupby(col)[TARGET].agg(["mean", "size"])
        g.columns = ["positive_rate", "n"]
        out[col.replace(" ", "_").lower()] = g.round(4).reset_index()
    return out


# ------------------------------------------------------------------ figure 6
def correlation(df: pd.DataFrame) -> pd.DataFrame:
    from config import COL_FEEDBACK
    num = df[[COL_AGE, COL_FEEDBACK, COL_RATING, COL_RECOMMEND, TARGET]].copy()
    num["review_words"] = df[TEXT_COL].str.split().str.len()
    corr = num.corr(numeric_only=True).round(3)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=.6,
                cbar_kws={"shrink": .8, "label": "Pearson r"}, ax=ax)
    ax.set_title("Correlation of numeric features")
    _save(fig, "fig06_correlation_heatmap.png")
    return corr.reset_index(names="feature")


# ------------------------------------------------------------------ figure 7
def wordclouds(df: pd.DataFrame) -> pd.DataFrame:
    from wordcloud import WordCloud, STOPWORDS
    stop = set(STOPWORDS) | {"retailer", "dress", "top", "shirt", "wear", "bought",
                             "ordered", "size", "fit", "fabric", "color"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    rows = []
    for lbl, ax, cmap in ((1, axes[0], "Greens"), (0, axes[1], "Reds")):
        text = " ".join(df.loc[df[TARGET] == lbl, TEXT_COL].str.lower())
        wc = WordCloud(width=1000, height=560, background_color="white",
                       stopwords=stop, colormap=cmap, max_words=120,
                       random_state=42).generate(text)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"{CLASS_NAMES[lbl].capitalize()} reviews", fontsize=17)
        for w, f in list(wc.words_.items())[:25]:
            rows.append({"class": CLASS_NAMES[lbl], "word": w, "relative_freq": round(f, 4)})
    _save(fig, "fig07_wordclouds.png")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure 8
def discriminative_terms(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Log-odds ratio terms - which words actually separate the classes."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(min_df=10, stop_words="english", ngram_range=(1, 2))
    X = cv.fit_transform(df[TEXT_COL])
    vocab = np.array(cv.get_feature_names_out())
    y = df[TARGET].values
    pos = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1
    neg = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1
    lo = np.log((pos / pos.sum()) / (neg / neg.sum()))
    order = np.argsort(lo)
    tbl = pd.concat([
        pd.DataFrame({"term": vocab[order[:top_n]], "log_odds": lo[order[:top_n]],
                      "direction": "negative"}),
        pd.DataFrame({"term": vocab[order[-top_n:]], "log_odds": lo[order[-top_n:]],
                      "direction": "positive"}),
    ]).round(3).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    sub = pd.concat([tbl[tbl.direction == "negative"],
                     tbl[tbl.direction == "positive"]]).sort_values("log_odds")
    ax.barh(sub["term"], sub["log_odds"],
            color=[NEG if d == "negative" else POS for d in sub["direction"]],
            edgecolor="black", lw=.5)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("log-odds ratio  (negative ← → positive)")
    ax.set_title(f"Top {top_n} discriminative terms per class")
    _save(fig, "fig08_discriminative_terms.png")
    return tbl
