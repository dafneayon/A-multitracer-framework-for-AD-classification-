"""
statics_1.py
==========
Mann-Whitney U Tests — Unified model (per-tracer subset) vs corresponding baseline.

Compares:
  - Unified model (FDG subset metrics)  vs  FDG-only baseline
  - Unified model (AMY subset metrics)  vs  AMY-only baseline

The test is **unpaired** (two-sided) for the same reason as statics.py:
splits are generated independently over different subject populations, so
the same seed does NOT guarantee identical test sets across models.

Usage:
    python statics_1.py \\
        --unified  summary_unified.csv \\
        --fdg      summary_fdg.csv \\
        --amy      summary_amy.csv

Optional:
    --exclude_seeds 61
    --output    mann_whitney_subset_results.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats

# Column name templates:
#   col_uni_tmpl : format with tracer name to get unified's per-tracer column
#   col_base     : column name in the single-tracer baseline summary
METRICS_SUBSET = {
    "Accuracy":    ("test_{}_acc",         "test_acc"),
    "AUROC":       ("test_{}_auroc",        "test_auroc"),
    "Sensitivity": ("test_{}_sensitivity",  "test_sensitivity"),
    "Specificity": ("test_{}_specificity",  "test_specificity"),
}


def load_summary(path: str, exclude_seeds=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if exclude_seeds:
        df = df[~df["seed"].isin(exclude_seeds)]
    return df.reset_index(drop=True)


def run_mann_whitney_subset(df_unified: pd.DataFrame,
                            df_baseline: pd.DataFrame,
                            tracer: str,
                            label_baseline: str,
                            all_results: list):
    """
    Two-sided Mann-Whitney U test comparing the unified model's per-tracer
    subset metrics against the corresponding single-tracer baseline.

    Args:
        df_unified    : Summary DataFrame for the unified model.
        df_baseline   : Summary DataFrame for the single-tracer baseline.
        tracer        : 'fdg' or 'amy'.
        label_baseline: Human-readable name for the baseline model.
        all_results   : List to accumulate result dicts for CSV export.
    """
    n_a = len(df_unified)
    n_b = len(df_baseline)

    print(f"\n  ── Unified ({tracer.upper()} subset) (n={n_a}) "
          f"vs {label_baseline} (n={n_b}) ──")
    print(f"  {'Metric':<14} {'Delta med.':>10}  {'95% CI':>22}  "
          f"{'p-value':>8}  {'r':>5}  Result")
    print(f"  {'─'*14}  {'─'*10}  {'─'*22}  {'─'*8}  {'─'*5}  {'─'*20}")

    for name, (col_uni_tmpl, col_base) in METRICS_SUBSET.items():
        col_uni = col_uni_tmpl.format(tracer.lower())

        if col_uni not in df_unified.columns:
            print(f"  {name:<14}  column '{col_uni}' not found in unified summary")
            continue
        if col_base not in df_baseline.columns:
            print(f"  {name:<14}  column '{col_base}' not found in baseline summary")
            continue

        a = df_unified[col_uni].values.astype(float)
        b = df_baseline[col_base].values.astype(float)

        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p = 1.0

        # Effect size r = |Z| / sqrt(N1 + N2)
        z = abs(stats.norm.ppf(p / 2)) if p < 1.0 else 0.0
        r = z / np.sqrt(n_a + n_b)

        # Median delta + bootstrap 95% CI
        delta_median = np.median(a) - np.median(b)
        np.random.seed(0)
        boot = [
            np.median(a[np.random.choice(n_a, n_a, replace=True)]) -
            np.median(b[np.random.choice(n_b, n_b, replace=True)])
            for _ in range(10_000)
        ]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        sig = ("sig. (p<0.05)"      if p < 0.05
               else "marginal (p<0.10)" if p < 0.10
               else "NS")
        mag = ("small"  if r < 0.3
               else "medium" if r < 0.5
               else "large")

        print(f"  {name:<14} {delta_median:>+10.4f}  "
              f"[{ci_lo:>+8.4f}, {ci_hi:>+7.4f}]  "
              f"{p:>8.4f}  {r:>5.2f}  {sig} ({mag})")

        all_results.append({
            "comparison":       f"Unified ({tracer}) vs {label_baseline}",
            "metric":           name,
            "n_unified":        n_a,
            "n_baseline":       n_b,
            "delta_medians":    round(delta_median, 4),
            "ci_95_low":        round(ci_lo, 4),
            "ci_95_high":       round(ci_hi, 4),
            "p_value":          round(p, 4),
            "effect_size_r":    round(r, 3),
            "significant":      p < 0.05,
            "effect_magnitude": mag,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified",       required=True,
                        help="Path to summary_unified.csv")
    parser.add_argument("--fdg",           required=True,
                        help="Path to summary_fdg.csv")
    parser.add_argument("--amy",           required=True,
                        help="Path to summary_amy.csv")
    parser.add_argument("--exclude_seeds", nargs="*", type=int, default=None,
                        help="Seed values to exclude from analysis")
    parser.add_argument("--output",        default="mann_whitney_subset_results.csv",
                        help="Output CSV path for results table")
    args = parser.parse_args()

    print("=" * 70)
    print("  MANN-WHITNEY U TEST — Unified (per-tracer) vs Baseline")
    print("  Unified FDG subset vs FDG-only baseline")
    print("  Unified AMY subset vs AMY-only baseline")
    print("  Unpaired test — splits generated over different subject populations")
    print("=" * 70)

    unified = load_summary(args.unified, args.exclude_seeds)
    fdg     = load_summary(args.fdg,     args.exclude_seeds)
    amy     = load_summary(args.amy,     args.exclude_seeds)

    excl = f" (excluded seeds: {args.exclude_seeds})" if args.exclude_seeds else ""
    print(f"\n  Iterations: Unified={len(unified)} | "
          f"FDG={len(fdg)} | AMY={len(amy)}{excl}")

    all_results = []

    print("\n" + "=" * 70)
    print("  STATISTICAL TESTS")
    print("=" * 70)

    run_mann_whitney_subset(unified, fdg, "fdg", "FDG baseline", all_results)
    run_mann_whitney_subset(unified, amy, "amy", "AMY baseline", all_results)

    pd.DataFrame(all_results).to_csv(args.output, index=False)

    print(f"""
{'─'*70}
  INTERPRETATION

  Answers: when the unified model processes FDG/AMY scans, does it
  outperform the specialist trained only on that tracer?

  Delta > 0  →  Unified outperforms the baseline (by median)
  95% CI excludes 0  →  consistent difference
  p < 0.05   →  statistically significant

  Why unpaired Mann-Whitney (not paired Wilcoxon):
    FDG and AMY splits are generated independently from the unified
    splits. Even though the same seed is used, the subject populations
    differ, so A[i] and B[i] do NOT share the same source of variance.
    Pairing requires identical test sets, which is not the case here.

  Results saved to: {args.output}
{'─'*70}
""")


if __name__ == "__main__":
    main()
