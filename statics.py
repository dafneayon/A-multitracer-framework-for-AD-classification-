"""
statics.py
========
Mann-Whitney U Tests — Unified model vs single-tracer baselines.

Compares the unified model against FDG-only and AMY-only baselines on the
aggregate test set and on each tracer-specific subset.

The test is **unpaired** (two-sided) because, although the same base seed is
used for all models, splits are drawn from different subject populations:
  - unified   : all subjects (FDG + AMY)
  - FDG/AMY   : each tracer's own subject subset

Consequently, the test sets are NOT identical across models for the same seed,
which invalidates the pairing assumption required by the Wilcoxon signed-rank
test.

Effect size : r = |Z| / sqrt(N1 + N2)
CI          : 95 % bootstrap on the difference of medians (10 000 resamples)

Usage:
    python statics.py \\
        --unified  summary_unified.csv \\
        --fdg      summary_fdg.csv \\
        --amy      summary_amy.csv

Optional:
    --exclude_seeds 61
    --output    mann_whitney_results.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats

# ── Metric column maps ────────────────────────────────────────────────────────
METRICS_AGGREGATE = {
    "Accuracy (aggregate)":    "test_acc",
    "AUROC (aggregate)":       "test_auroc",
    "Sensitivity (aggregate)": "test_sensitivity",
    "Specificity (aggregate)": "test_specificity",
}

METRICS_FDG = {
    "Accuracy (FDG)":    "test_fdg_acc",
    "AUROC (FDG)":       "test_fdg_auroc",
    "Sensitivity (FDG)": "test_fdg_sensitivity",
    "Specificity (FDG)": "test_fdg_specificity",
}

METRICS_AMY = {
    "Accuracy (AMY)":    "test_amy_acc",
    "AUROC (AMY)":       "test_amy_auroc",
    "Sensitivity (AMY)": "test_amy_sensitivity",
    "Specificity (AMY)": "test_amy_specificity",
}


def load_summary(path: str, exclude_seeds=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "seed" in df.columns, f"Column 'seed' not found in {path}"
    if exclude_seeds:
        before = len(df)
        df = df[~df["seed"].isin(exclude_seeds)]
        print(f"  [{path}] Excluded {before - len(df)} iterations "
              f"(seeds: {exclude_seeds})")
    return df.set_index("seed").sort_index()


def print_descriptive(df: pd.DataFrame, label: str, metrics: dict):
    print(f"\n  {label}")
    print(f"  {'Metric':<28} {'Mean':>8}  {'SD':>8}  {'Median':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}")
    for name, col in metrics.items():
        if col in df.columns:
            print(f"  {name:<28} {df[col].mean():>8.4f}  "
                  f"{df[col].std():>8.4f}  {df[col].median():>8.4f}")


def run_mann_whitney(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     label_a: str, label_b: str,
                     metrics: dict, all_results: list):
    """
    Two-sided Mann-Whitney U test (unpaired).

    Delta > 0  →  model A outperforms model B (by median).
    Effect size r = |Z| / sqrt(N1 + N2).
    95 % CI via bootstrap on the difference of medians.
    """
    n_a, n_b = len(df_a), len(df_b)

    print(f"\n  ── {label_a} (n={n_a}) vs {label_b} (n={n_b}) ──")
    print(f"  {'Metric':<28} {'Delta med.':>10}  "
          f"{'95% CI':>22}  {'p-value':>8}  {'r':>5}  Result")
    print(f"  {'─'*28}  {'─'*10}  {'─'*22}  {'─'*8}  {'─'*5}  {'─'*20}")

    for name, col in metrics.items():
        if col not in df_a.columns or col not in df_b.columns:
            continue

        a = df_a[col].values.astype(float)
        b = df_b[col].values.astype(float)

        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p = 1.0

        # Effect size
        z = abs(stats.norm.ppf(p / 2)) if p < 1.0 else 0.0
        r = z / np.sqrt(n_a + n_b)

        # Median delta + bootstrap CI
        delta_median = np.median(a) - np.median(b)
        delta_mean   = np.mean(a) - np.mean(b)

        np.random.seed(0)
        boot = [
            np.median(a[np.random.choice(n_a, n_a, replace=True)]) -
            np.median(b[np.random.choice(n_b, n_b, replace=True)])
            for _ in range(10_000)
        ]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        sig = ("sig. (p<0.05)"   if p < 0.05
               else "marginal (p<0.10)" if p < 0.10
               else "NS")
        mag = ("small"  if r < 0.3
               else "medium" if r < 0.5
               else "large")

        print(f"  {name:<28} {delta_median:>+10.4f}  "
              f"[{ci_lo:>+8.4f}, {ci_hi:>+7.4f}]  "
              f"{p:>8.4f}  {r:>5.2f}  {sig} ({mag})")

        all_results.append({
            "comparison":        f"{label_a} vs {label_b}",
            "metric":            name,
            "n_a":               n_a,
            "n_b":               n_b,
            "delta_medians":     round(delta_median, 4),
            "delta_means":       round(delta_mean, 4),
            "ci_95_low":         round(ci_lo, 4),
            "ci_95_high":        round(ci_hi, 4),
            "p_value":           round(p, 4),
            "effect_size_r":     round(r, 3),
            "significant":       p < 0.05,
            "effect_magnitude":  mag,
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
    parser.add_argument("--output",        default="mann_whitney_results.csv",
                        help="Output CSV path for results table")
    args = parser.parse_args()

    print("=" * 75)
    print("  MANN-WHITNEY U TEST (unpaired, two-sided)")
    print("  Unified vs FDG baseline  |  Unified vs AMY baseline")
    print("  Rationale: same seed but test sets are NOT identical")
    print("  (unified stratifies over all subjects;")
    print("   FDG/AMY stratify over their own subject subsets)")
    print("  Effect size r = |Z| / sqrt(N1+N2)")
    print("  95% CI: bootstrap on difference of medians (10 000 resamples)")
    print("=" * 75)

    unified = load_summary(args.unified, args.exclude_seeds)
    fdg     = load_summary(args.fdg,     args.exclude_seeds)
    amy     = load_summary(args.amy,     args.exclude_seeds)

    print(f"\n  Iterations loaded — Unified: {len(unified)} | "
          f"FDG: {len(fdg)} | AMY: {len(amy)}")

    # ── Descriptive statistics ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  DESCRIPTIVE STATISTICS (test set)")
    print("=" * 75)
    print_descriptive(unified, "Unified model",  METRICS_AGGREGATE)
    print_descriptive(fdg,     "FDG baseline",   {**METRICS_AGGREGATE, **METRICS_FDG})
    print_descriptive(amy,     "AMY baseline",   {**METRICS_AGGREGATE, **METRICS_AMY})

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  STATISTICAL TESTS — Mann-Whitney U (two-sided)")
    print("=" * 75)

    all_results = []

    run_mann_whitney(unified, fdg,
                     "Unified (aggregate)", "FDG baseline",
                     METRICS_AGGREGATE, all_results)

    run_mann_whitney(unified, fdg,
                     "Unified (FDG subset)", "FDG baseline",
                     METRICS_FDG, all_results)

    run_mann_whitney(unified, amy,
                     "Unified (aggregate)", "AMY baseline",
                     METRICS_AGGREGATE, all_results)

    run_mann_whitney(unified, amy,
                     "Unified (AMY subset)", "AMY baseline",
                     METRICS_AMY, all_results)

    pd.DataFrame(all_results).to_csv(args.output, index=False)

    print(f"""
{'=' * 75}
  INTERPRETATION GUIDE

  Delta > 0  →  Unified outperforms the baseline (by median)
  Delta < 0  →  Baseline outperforms the unified model
  95% CI excludes 0  →  consistent difference

  p < 0.05          →  statistically significant
  0.05 ≤ p < 0.10   →  marginal trend
  p ≥ 0.10          →  no evidence of difference

  Effect size r = |Z| / sqrt(N1 + N2):
    < 0.30   →  small
    0.30–0.50  →  medium
    > 0.50   →  large

  Results saved to: {args.output}
{'=' * 75}
""")


if __name__ == "__main__":
    main()
