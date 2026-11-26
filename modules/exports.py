"""Excel exports - one workbook per stage, plus the consolidated paper tables."""
from __future__ import annotations

import pandas as pd

from config import RESULTS, excel


def write(sheets: dict[str, pd.DataFrame], filename: str) -> str:
    """Write a dict of frames to one workbook, one sheet each."""
    path = excel(filename)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, frame in sheets.items():
            if frame is None or len(frame) == 0:
                continue
            frame.to_excel(xl, sheet_name=name[:31], index=False)
    print(f"   wrote {filename}  ({len(sheets)} sheets)")
    return str(path)


def paper_tables(final: pd.DataFrame, tune_summary: pd.DataFrame,
                 sig: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The exact tables that go into the paper, pre-formatted."""
    tuned = final[final.stage == "tuned"].copy()
    base = final[final.stage == "baseline"].copy()

    # Table I - main comparison
    t1 = (pd.concat([tuned, base])
          [["model", "family", "test_accuracy", "test_macro_f1", "test_mcc",
            "test_cohen_kappa", "test_roc_auc", "test_pr_auc"]]
          .sort_values("test_macro_f1", ascending=False)
          .rename(columns={"test_accuracy": "Accuracy", "test_macro_f1": "Macro-F1",
                           "test_mcc": "MCC", "test_cohen_kappa": "Cohen kappa",
                           "test_roc_auc": "ROC-AUC", "test_pr_auc": "PR-AUC",
                           "model": "Model", "family": "Family"})
          .round(4).reset_index(drop=True))

    # Table II - per-class detail
    t2 = (tuned[["model", "test_precision_negative", "test_recall_negative",
                 "test_f1_negative", "test_precision_positive",
                 "test_recall_positive", "test_tn", "test_fp", "test_fn", "test_tp"]]
          .rename(columns=lambda c: c.replace("test_", ""))
          .sort_values("recall_negative", ascending=False)
          .round(4).reset_index(drop=True))

    # Table III - untuned vs tuned lift
    piv = (final[final.stage.isin(["untuned", "tuned"])]
           .pivot_table(index="model", columns="stage",
                        values=["test_accuracy", "test_macro_f1", "test_mcc"]))
    piv.columns = [f"{b}_{a.replace('test_', '')}" for a, b in piv.columns]
    t3 = piv.reset_index()
    for m in ("accuracy", "macro_f1", "mcc"):
        if f"tuned_{m}" in t3 and f"untuned_{m}" in t3:
            t3[f"lift_{m}"] = (t3[f"tuned_{m}"] - t3[f"untuned_{m}"]).round(4)
    t3 = t3.round(4)

    # Table IV - runtime
    t4 = (tuned[["model", "family", "representation", "fit_seconds", "params"]]
          .sort_values("fit_seconds").reset_index(drop=True))

    return {"TableI_main_comparison": t1, "TableII_per_class": t2,
            "TableIII_tuning_lift": t3, "TableIV_runtime_params": t4,
            "TableV_significance": sig, "tuning_summary": tune_summary}


def summary_text(final: pd.DataFrame) -> str:
    """Console summary block, mirroring the depression notebook."""
    tuned = final[final.stage == "tuned"]
    base = final[final.model == "Baseline (majority)"].iloc[0]
    best = tuned.sort_values("test_macro_f1", ascending=False).iloc[0]
    over95 = tuned[tuned.test_accuracy >= 0.95]
    lines = [
        "=" * 68, "SUMMARY", "=" * 68,
        f"models trained            : {len(tuned)}",
        f"majority baseline accuracy: {base.test_accuracy:.4f}",
        f"majority baseline macro-F1: {base.test_macro_f1:.4f}",
        f"best model                : {best.model}",
        f"  accuracy                : {best.test_accuracy:.4f}",
        f"  macro-F1                : {best.test_macro_f1:.4f}",
        f"  MCC                     : {best.test_mcc:.4f}",
        f"  ROC-AUC                 : {best.test_roc_auc:.4f}",
        f"  negative-class recall   : {best.test_recall_negative:.4f}",
        f"models at or above 95%    : {len(over95)}  ({', '.join(over95.model)})",
        f"accuracy gain over baseline: {(best.test_accuracy - base.test_accuracy)*100:+.2f} pts",
        f"macro-F1 gain over baseline: {best.test_macro_f1 - base.test_macro_f1:+.4f}",
        "=" * 68,
    ]
    return "\n".join(lines)
