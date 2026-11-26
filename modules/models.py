"""Model registry, metric block and the training loop.

Each entry declares which representation it consumes ("sparse" or "dense") so
the training loop can hand every model the matrix it is actually suited to.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from config import SEED


# ------------------------------------------------------------------ registry
@dataclass
class Spec:
    """One model: how to build it, and which matrix it eats."""
    name: str
    family: str
    representation: str          # "sparse" or "dense"
    build: Callable[..., Any]
    tunable: bool = True


def _xgb(**kw):
    from xgboost import XGBClassifier
    return XGBClassifier(tree_method="hist", eval_metric="logloss",
                         random_state=SEED, n_jobs=-1, **kw)


def _lgbm(**kw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **kw)


def _cat(**kw):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False, **kw)


REGISTRY: list[Spec] = [
    Spec("Logistic Regression", "Linear", "sparse",
         lambda **kw: LogisticRegression(max_iter=3000, random_state=SEED, **kw)),
    Spec("Linear SVM", "Linear", "sparse",
         lambda **kw: CalibratedClassifierCV(
             LinearSVC(max_iter=5000, random_state=SEED, **kw), cv=3)),
    Spec("Multinomial NB", "Probabilistic", "sparse",
         lambda **kw: MultinomialNB(**kw)),
    Spec("k-Nearest Neighbours", "Instance-based", "dense",
         lambda **kw: KNeighborsClassifier(n_jobs=-1, **kw)),
    Spec("Random Forest", "Tree", "dense",
         lambda **kw: RandomForestClassifier(random_state=SEED, n_jobs=-1, **kw)),
    Spec("XGBoost", "Boosting", "dense", _xgb),
    Spec("LightGBM", "Boosting", "dense", _lgbm),
    Spec("CatBoost", "Boosting", "dense", _cat),
    Spec("Extra Trees", "Tree", "dense",
         lambda **kw: ExtraTreesClassifier(random_state=SEED, n_jobs=-1, **kw)),
    Spec("MLP (Neural Net)", "Neural", "dense",
         lambda **kw: MLPClassifier(random_state=SEED, early_stopping=True,
                                    n_iter_no_change=8, **kw)),
]

BASELINES: list[Spec] = [
    Spec("Baseline (majority)", "Baseline", "dense",
         lambda **kw: DummyClassifier(strategy="most_frequent"), tunable=False),
    Spec("Baseline (stratified)", "Baseline", "dense",
         lambda **kw: DummyClassifier(strategy="stratified", random_state=SEED),
         tunable=False),
]

# default hyper-parameters used for the untuned pass
DEFAULTS: dict[str, dict] = {
    "Logistic Regression": dict(C=8.0),
    "Linear SVM": dict(C=0.5),
    "Multinomial NB": dict(alpha=0.1),
    "k-Nearest Neighbours": dict(n_neighbors=25, weights="distance", metric="cosine"),
    "Random Forest": dict(n_estimators=300, min_samples_leaf=2),
    "XGBoost": dict(n_estimators=400, learning_rate=0.1, max_depth=6),
    "LightGBM": dict(n_estimators=400, learning_rate=0.1, num_leaves=63),
    "CatBoost": dict(iterations=400, depth=6, learning_rate=0.1),
    "Extra Trees": dict(n_estimators=400, min_samples_leaf=2),
    "MLP (Neural Net)": dict(hidden_layer_sizes=(256,), alpha=1e-4, max_iter=60),
}


# ------------------------------------------------------------------ metrics
def scores(model, X) -> np.ndarray:
    """Positive-class score, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def metric_block(y_true, y_pred, y_score=None) -> dict[str, float]:
    """The full metric set from WORKFLOW.md, macro-F1 first."""
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "recall_negative": recall_score(y_true, y_pred, pos_label=0),
        "precision_negative": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_negative": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, pos_label=1),
        "precision_positive": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
            out["pr_auc"] = average_precision_score(y_true, y_score)
        except ValueError:
            out["roc_auc"] = out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


# ------------------------------------------------------------------ training
def matrices(spec: Spec, feats) -> tuple:
    """Pick the representation this model was declared against."""
    src = feats.sparse if spec.representation == "sparse" else feats.dense
    return src["train"], src["val"], src["test"]


def fit_and_score(spec: Spec, feats, split, params: dict | None = None,
                  stage: str = "default") -> tuple[dict, Any]:
    """Fit one model and evaluate it on val and test."""
    A, V, B = matrices(spec, feats)
    params = params if params is not None else DEFAULTS.get(spec.name, {})

    t0 = time.time()
    model = spec.build(**params)
    model.fit(A, split.y_train)
    fit_s = time.time() - t0

    val = metric_block(split.y_val, model.predict(V), scores(model, V))
    test = metric_block(split.y_test, model.predict(B), scores(model, B))

    row = {"model": spec.name, "family": spec.family, "stage": stage,
           "representation": spec.representation, "fit_seconds": round(fit_s, 2),
           "params": str(params)}
    row.update({f"val_{k}": v for k, v in val.items()})
    row.update({f"test_{k}": v for k, v in test.items()})
    return row, model


def run_all(specs: list[Spec], feats, split, stage: str = "default",
            params_by_model: dict[str, dict] | None = None,
            verbose: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit every spec in turn, returning a tidy results frame."""
    rows, fitted = [], {}
    for spec in specs:
        p = (params_by_model or {}).get(spec.name)
        row, model = fit_and_score(spec, feats, split, p, stage)
        rows.append(row)
        fitted[spec.name] = model
        if verbose:
            print(f"  {spec.name:22s} acc={row['test_accuracy']:.4f}  "
                  f"macroF1={row['test_macro_f1']:.4f}  MCC={row['test_mcc']:.4f}  "
                  f"[{row['fit_seconds']:.1f}s]", flush=True)
    return pd.DataFrame(rows), fitted


def leaderboard(results: pd.DataFrame, by: str = "test_macro_f1") -> pd.DataFrame:
    """Presentation view: the columns that go into the paper table."""
    cols = ["model", "family", "stage", "test_accuracy", "test_macro_f1", "test_mcc",
            "test_cohen_kappa", "test_roc_auc", "test_pr_auc",
            "test_recall_negative", "test_precision_negative", "fit_seconds"]
    keep = [c for c in cols if c in results.columns]
    return (results[keep].sort_values(by, ascending=False)
            .round(4).reset_index(drop=True))
