"""Optuna hyper-parameter search.

Every study optimises macro-F1 on the validation split, never accuracy - on an
88% -imbalanced target accuracy rewards a model for ignoring the minority class.

Trial budgets are per-model because a CatBoost fit costs ~25s while a
MultinomialNB fit costs ~0.02s; a flat budget would spend all the time in one
place and starve the rest.
"""
from __future__ import annotations

import time
from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import f1_score

from config import SEED
from models import REGISTRY, matrices

optuna.logging.set_verbosity(optuna.logging.WARNING)

# per-model trial budgets, chosen from the measured untuned fit times
TRIALS: dict[str, int] = {
    "Logistic Regression": 30,
    "Linear SVM": 25,
    "Multinomial NB": 30,
    "k-Nearest Neighbours": 15,
    "Random Forest": 15,
    "XGBoost": 15,
    "LightGBM": 15,
    "CatBoost": 12,
    "Extra Trees": 15,
    "MLP (Neural Net)": 12,
}


def space(name: str, t: optuna.Trial) -> dict[str, Any]:
    """Search space per model. class_weight is in every space that supports it -
    it is the main lever against the 88/12 imbalance."""
    if name == "Logistic Regression":
        return dict(
            C=t.suggest_float("C", 0.05, 50, log=True),
            class_weight=t.suggest_categorical("class_weight", [None, "balanced"]),
        )
    if name == "Linear SVM":
        return dict(
            C=t.suggest_float("C", 0.01, 10, log=True),
            class_weight=t.suggest_categorical("class_weight", [None, "balanced"]),
        )
    if name == "Multinomial NB":
        return dict(alpha=t.suggest_float("alpha", 0.005, 3.0, log=True))
    if name == "k-Nearest Neighbours":
        return dict(
            n_neighbors=t.suggest_int("n_neighbors", 5, 75, step=5),
            weights=t.suggest_categorical("weights", ["uniform", "distance"]),
            metric=t.suggest_categorical("metric", ["cosine", "euclidean"]),
        )
    if name == "Random Forest":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 700, step=100),
            max_depth=t.suggest_categorical("max_depth", [None, 20, 40, 60]),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8),
            max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            class_weight=t.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]),
        )
    if name == "XGBoost":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_depth=t.suggest_int("max_depth", 3, 10),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            scale_pos_weight=t.suggest_float("scale_pos_weight", 0.1, 1.5),
        )
    if name == "LightGBM":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
            num_leaves=t.suggest_int("num_leaves", 15, 127, step=16),
            min_child_samples=t.suggest_int("min_child_samples", 5, 60),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            class_weight=t.suggest_categorical("class_weight", [None, "balanced"]),
        )
    if name == "CatBoost":
        return dict(
            iterations=t.suggest_int("iterations", 200, 700, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
            depth=t.suggest_int("depth", 4, 9),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 12.0),
            auto_class_weights=t.suggest_categorical(
                "auto_class_weights", [None, "Balanced"]),
        )
    if name == "Extra Trees":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 700, step=100),
            max_depth=t.suggest_categorical("max_depth", [None, 20, 40, 60]),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8),
            max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            class_weight=t.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]),
        )
    if name == "MLP (Neural Net)":
        return dict(
            hidden_layer_sizes=t.suggest_categorical(
                "hidden_layer_sizes", [(128,), (256,), (512,), (256, 128), (512, 128)]),
            alpha=t.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            batch_size=t.suggest_categorical("batch_size", [128, 256]),
            max_iter=t.suggest_int("max_iter", 40, 120, step=20),
        )
    return {}


def tune_one(spec, feats, split, n_trials: int | None = None
             ) -> tuple[dict, optuna.Study, float]:
    """Run one study, returning the best params and the val macro-F1 it reached."""
    A, V, _ = matrices(spec, feats)
    n = n_trials if n_trials is not None else TRIALS.get(spec.name, 15)

    def objective(t: optuna.Trial) -> float:
        params = space(spec.name, t)
        model = spec.build(**params)
        model.fit(A, split.y_train)
        return f1_score(split.y_val, model.predict(V), average="macro")

    study = optuna.create_study(
        direction="maximize", study_name=spec.name,
        sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    return study.best_params, study, time.time() - t0


def tune_all(feats, split, specs=None, verbose: bool = True
             ) -> tuple[dict[str, dict], dict[str, optuna.Study], pd.DataFrame]:
    """Tune every model, reporting the lift over its untuned validation score."""
    specs = specs or REGISTRY
    best, studies, rows = {}, {}, []
    for spec in specs:
        n = TRIALS.get(spec.name, 15)
        if verbose:
            print(f"  tuning {spec.name:22s} ({n} trials) ...", end="", flush=True)
        params, study, secs = tune_one(spec, feats, split, n)
        best[spec.name] = params
        studies[spec.name] = study
        rows.append({"model": spec.name, "n_trials": n,
                     "best_val_macro_f1": round(study.best_value, 4),
                     "seconds": round(secs, 1), "best_params": str(params)})
        if verbose:
            print(f" val macroF1={study.best_value:.4f}  [{secs:.0f}s]", flush=True)
    return best, studies, pd.DataFrame(rows)


def history(studies: dict[str, optuna.Study]) -> pd.DataFrame:
    """Every trial from every study, flattened for the Excel export."""
    frames = []
    for name, st in studies.items():
        d = st.trials_dataframe(attrs=("number", "value", "params", "duration"))
        d.insert(0, "model", name)
        d["running_best"] = d["value"].cummax()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
