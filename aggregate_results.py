"""
aggregate_results.py
--------------------
Runs ONCE after all SLURM array jobs finish.
Reads the 25 individual result CSVs per model and produces:

  1. summary_{model}.csv  — full table (one row per iteration)
  2. Console report in the format described in the paper:
       "Results are reported as mean ± standard deviation of
        accuracy, validation loss, and AUROC across iterations."

Usage:
  python aggregate_results.py \
      --base_dir /path/to/experiments \
      --expname_unified exp_unified \
      --expname_fdg     exp_fdg \
      --expname_amy     exp_amy \
      --n_iterations 25
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_results(workdir: str, prefix: str, n_iterations: int) -> pd.DataFrame:
    frames  = []
    missing = []
    for i in range(1, n_iterations + 1):
        path = os.path.join(workdir, "results", f"result_{prefix}_iter{i:02d}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            missing.append(i)

    if missing:
        print(f"  ⚠  Missing iterations for '{prefix}': {missing}")
    if not frames:
        raise FileNotFoundError(
            f"No result files found for prefix '{prefix}' in {workdir}/results/"
        )
    return pd.concat(frames, ignore_index=True)


def mean_std(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return "N/A"
    return f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}"


def report_block(df: pd.DataFrame, title: str, metrics: List[dict]):
    """
    metrics = list of {"col": str, "label": str}
    """
    n = len(df)
    print(f"\n{'─'*60}")
    print(f"  {title}   (n={n} iterations)")
    print(f"{'─'*60}")
    for m in metrics:
        col   = m["col"]
        label = m["label"]
        if col not in df.columns:
            print(f"  {label:45s}  —  (column not found)")
            continue
        print(f"  {label:45s}  {mean_std(df[col])}")


# ──────────────────────────────────────────────
# Metric definitions  (paper-reported first)
# ──────────────────────────────────────────────

# Paper reports: accuracy, validation loss, AUROC
PAPER_METRICS_UNIFIED = [
    {"col": "val_loss",          "label": "Validation Loss"},
    {"col": "val_acc",           "label": "Validation Accuracy"},
    {"col": "val_auroc",         "label": "Validation AUROC"},
    {"col": "test_acc",          "label": "Test Accuracy  (aggregate)"},
    {"col": "test_auroc",        "label": "Test AUROC     (aggregate)"},
    {"col": "test_sensitivity",  "label": "Test Sensitivity (aggregate)"},
    {"col": "test_specificity",  "label": "Test Specificity (aggregate)"},
]

PAPER_METRICS_UNIFIED_FDG = [
    {"col": "test_fdg_acc",         "label": "Test Accuracy   (FDG subset)"},
    {"col": "test_fdg_auroc",       "label": "Test AUROC      (FDG subset)"},
    {"col": "test_fdg_sensitivity", "label": "Test Sensitivity (FDG subset)"},
    {"col": "test_fdg_specificity", "label": "Test Specificity (FDG subset)"},
]

PAPER_METRICS_UNIFIED_AMY = [
    {"col": "test_amy_acc",         "label": "Test Accuracy   (AMY subset)"},
    {"col": "test_amy_auroc",       "label": "Test AUROC      (AMY subset)"},
    {"col": "test_amy_sensitivity", "label": "Test Sensitivity (AMY subset)"},
    {"col": "test_amy_specificity", "label": "Test Specificity (AMY subset)"},
]

PAPER_METRICS_SINGLE = [
    {"col": "val_loss",         "label": "Validation Loss"},
    {"col": "val_acc",          "label": "Validation Accuracy"},
    {"col": "val_auroc",        "label": "Validation AUROC"},
    {"col": "test_acc",         "label": "Test Accuracy"},
    {"col": "test_auroc",       "label": "Test AUROC"},
    {"col": "test_sensitivity", "label": "Test Sensitivity"},
    {"col": "test_specificity", "label": "Test Specificity"},
]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",        type=str, required=True)
    parser.add_argument("--expname_unified", type=str, default="exp_unified")
    parser.add_argument("--expname_fdg",     type=str, default="exp_fdg")
    parser.add_argument("--expname_amy",     type=str, default="exp_amy")
    parser.add_argument("--n_iterations",    type=int, default=25)
    args = parser.parse_args()

    configs = [
        ("unified", args.expname_unified, "unified"),
        ("fdg",     args.expname_fdg,     "fdg"),
        ("amy",     args.expname_amy,     "amy"),
    ]

    summaries = {}

    for model_key, expname, prefix in configs:
        workdir = os.path.join(args.base_dir, expname)
        print(f"\nLoading {model_key.upper()} results from: {workdir}/results/")

        try:
            df = load_results(workdir, prefix, args.n_iterations)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        summaries[model_key] = df

        # Save full per-iteration table
        out_path = os.path.join(workdir, f"summary_{model_key}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved full summary → {out_path}")

    # ── Console report (paper format) ─────────
    print(f"\n\n{'='*60}")
    print("  REPEATED RANDOM-SPLIT STABILITY ANALYSIS")
    print("  Reported as mean ± std across iterations")
    print(f"{'='*60}")

    if "unified" in summaries:
        df = summaries["unified"]
        report_block(df, "UNIFIED MODEL (FDG + AMY)", PAPER_METRICS_UNIFIED)
        report_block(df, "UNIFIED MODEL — FDG subset", PAPER_METRICS_UNIFIED_FDG)
        report_block(df, "UNIFIED MODEL — AMY subset", PAPER_METRICS_UNIFIED_AMY)

    if "fdg" in summaries:
        report_block(summaries["fdg"], "FDG UNIMODAL BASELINE", PAPER_METRICS_SINGLE)

    if "amy" in summaries:
        report_block(summaries["amy"], "AMY UNIMODAL BASELINE", PAPER_METRICS_SINGLE)

    # ── Cross-model comparison table ──────────
    print(f"\n{'='*60}")
    print("  CROSS-MODEL COMPARISON  (Test set, mean ± std)")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Unified':>22} {'FDG':>22} {'AMY':>22}")
    print(f"  {'-'*20} {'-'*22} {'-'*22} {'-'*22}")

    compare_metrics = [
        ("Test Accuracy",    "test_acc"),
        ("Test AUROC",       "test_auroc"),
        ("Test Sensitivity", "test_sensitivity"),
        ("Test Specificity", "test_specificity"),
        ("Val Loss",         "val_loss"),
        ("Val AUROC",        "val_auroc"),
    ]

    for label, col in compare_metrics:
        row = f"  {label:<20}"
        for key in ["unified", "fdg", "amy"]:
            if key in summaries and col in summaries[key].columns:
                row += f"  {mean_std(summaries[key][col]):>22}"
            else:
                row += f"  {'N/A':>22}"
        print(row)

    print(f"\n✓ Aggregation complete.\n")


if __name__ == "__main__":
    main()