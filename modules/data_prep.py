"""Loading and cleaning for the Women's E-Commerce Clothing Reviews dataset.

The download tries the local `dataset/` folder first so the notebook re-runs
offline, and falls back to kagglehub when the file is not there yet.
"""
from __future__ import annotations

import pandas as pd

from config import (CSV_NAME, DATASET_DIR, KAGGLE_SLUG, COL_TITLE, COL_TEXT,
                    COL_RATING, NEUTRAL_RATING, TEXT_COL, TARGET)


def download() -> "pd.DataFrame":
    """Return the raw dataframe, downloading from Kaggle only if needed."""
    local = DATASET_DIR / CSV_NAME
    if local.exists():
        print(f"[data] using local copy: {local.name}")
        return pd.read_csv(local, index_col=0)

    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_SLUG)
    print(f"[data] downloaded to: {path}")
    src = next(p for p in __import__("pathlib").Path(path).rglob("*.csv"))
    df = pd.read_csv(src, index_col=0)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(local)
    print(f"[data] cached at: {local}")
    return df


def profile(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Structured profile of the raw data, ready for an Excel export."""
    overview = pd.DataFrame({
        "metric": ["rows", "columns", "duplicated rows", "memory (MB)"],
        "value": [len(df), df.shape[1], int(df.duplicated().sum()),
                  round(df.memory_usage(deep=True).sum() / 1e6, 2)],
    })
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "nulls": df.isna().sum().values,
        "null_pct": (df.isna().mean().values * 100).round(2),
        "unique": [df[c].nunique() for c in df.columns],
    })
    describe = df.describe(include="all").T.reset_index(names="column")
    return {"overview": overview, "schema": schema, "describe": describe}


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply the cleaning contract from WORKFLOW.md.

    1. drop rows with no review body
    2. fill missing titles with an empty string
    3. build the combined text column
    4. drop the ambiguous neutral rating
    5. build the binary sentiment target
    """
    steps: list[tuple[str, int]] = [("raw", len(df))]
    out = df.copy()

    out = out[out[COL_TEXT].notna() & (out[COL_TEXT].str.strip() != "")]
    steps.append(("after dropping empty Review Text", len(out)))

    out[COL_TITLE] = out[COL_TITLE].fillna("")
    out[TEXT_COL] = (out[COL_TITLE].astype(str) + " " +
                     out[COL_TEXT].astype(str)).str.strip()

    out = out[out[COL_RATING] != NEUTRAL_RATING]
    steps.append((f"after dropping Rating == {NEUTRAL_RATING}", len(out)))

    # identical review bodies would otherwise leak across the train/test split
    out = out.drop_duplicates(subset=[TEXT_COL], keep="first")
    steps.append(("after dropping duplicate texts", len(out)))

    out[TARGET] = (out[COL_RATING] >= 4).astype(int)
    out = out.reset_index(drop=True)

    if verbose:
        print("[clean] row counts")
        for label, n in steps:
            print(f"         {label:38s} {n:6d}")
        pos = out[TARGET].mean()
        print(f"[clean] positive rate {pos:.4f}   negative rate {1 - pos:.4f}")

    return out


def neutral_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Rating x Recommended crosstab - the justification for dropping Rating 3.

    Built from the RAW frame, before the neutral class is removed.
    """
    from config import COL_RECOMMEND
    ct = pd.crosstab(df[COL_RATING], df[COL_RECOMMEND], normalize="index")
    ct.columns = ["not_recommended", "recommended"]
    ct["n"] = df[COL_RATING].value_counts().sort_index()
    ct["share_of_data"] = (ct["n"] / len(df))
    return ct.round(4).reset_index()


def cleaning_log(raw: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Before/after table for the Excel export."""
    has_text = raw[COL_TEXT].notna() & (raw[COL_TEXT].str.strip() != "")
    after_text = int(has_text.sum())
    after_neutral = int((has_text & (raw[COL_RATING] != NEUTRAL_RATING)).sum())
    return pd.DataFrame({
        "stage": ["raw", "empty text removed",
                  f"Rating=={NEUTRAL_RATING} removed", "duplicate texts removed"],
        "rows": [len(raw), after_text, after_neutral, len(clean_df)],
    }).assign(removed=lambda d: d["rows"].shift(1).sub(d["rows"]).fillna(0).astype(int))
