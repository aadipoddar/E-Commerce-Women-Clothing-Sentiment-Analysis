"""Train/val/test splitting and the two feature representations.

Sparse TF-IDF feeds the linear and probabilistic models; a TruncatedSVD
projection of the same matrix feeds the tree and boosting models, which cannot
work efficiently on a 100k-column sparse matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from config import (SEED, SVD_COMPONENTS, TARGET, TEST_SIZE, TEXT_COL,
                    TFIDF_CHAR, TFIDF_WORD, VAL_SIZE)


@dataclass
class Split:
    """Raw text splits plus their labels."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            y = getattr(self, f"y_{name}")
            rows.append({"split": name, "n": len(y),
                         "positive": int(y.sum()), "negative": int((1 - y).sum()),
                         "positive_rate": round(float(y.mean()), 4)})
        total = sum(r["n"] for r in rows)
        for r in rows:
            r["share_of_data"] = round(r["n"] / total, 4)
        return pd.DataFrame(rows)


@dataclass
class Features:
    """Fitted representations. Sparse for linear models, dense for trees."""
    sparse: dict[str, Any] = field(default_factory=dict)
    dense: dict[str, Any] = field(default_factory=dict)
    vectorizer: Any = None
    svd: Any = None
    scaler: Any = None

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"representation": "TF-IDF (word 1-2 + char_wb 3-5)",
             "n_features": self.sparse["train"].shape[1],
             "used_by": "LogReg, LinearSVM, MultinomialNB"},
            {"representation": f"TruncatedSVD ({SVD_COMPONENTS} comps, standardised)",
             "n_features": self.dense["train"].shape[1],
             "used_by": "RandomForest, XGBoost, LightGBM, CatBoost"},
        ])


def make_split(df: pd.DataFrame) -> Split:
    """Stratified 70/15/15 split with a fixed seed."""
    X, y = df[TEXT_COL].values, df[TARGET].values
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac, random_state=SEED, stratify=y_rest)
    print(f"[split] train {len(y_train)} / val {len(y_val)} / test {len(y_test)}")
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def build_features(split: Split, verbose: bool = True) -> Features:
    """Fit on train only, then transform val and test.

    Fitting the vectoriser on the training split alone is what keeps the
    evaluation honest - no test vocabulary leaks into the model.
    """
    vec = FeatureUnion([
        ("word", TfidfVectorizer(**TFIDF_WORD)),
        ("char", TfidfVectorizer(**TFIDF_CHAR)),
    ])
    A = vec.fit_transform(split.X_train)
    V = vec.transform(split.X_val)
    B = vec.transform(split.X_test)
    if verbose:
        print(f"[features] TF-IDF: {A.shape[0]} x {A.shape[1]}")

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
    As = svd.fit_transform(A)
    Vs, Bs = svd.transform(V), svd.transform(B)
    scaler = StandardScaler().fit(As)
    if verbose:
        evr = svd.explained_variance_ratio_.sum()
        print(f"[features] SVD-{SVD_COMPONENTS}: explained variance {evr:.4f}")

    return Features(
        sparse={"train": A, "val": V, "test": B},
        dense={"train": scaler.transform(As), "val": scaler.transform(Vs),
               "test": scaler.transform(Bs)},
        vectorizer=vec, svd=svd, scaler=scaler,
    )


def svd_variance_table(feats: Features) -> pd.DataFrame:
    """Cumulative explained variance, for the diagnostics workbook."""
    evr = feats.svd.explained_variance_ratio_
    return pd.DataFrame({
        "component": np.arange(1, len(evr) + 1),
        "explained_variance": evr.round(6),
        "cumulative": evr.cumsum().round(6),
    })
