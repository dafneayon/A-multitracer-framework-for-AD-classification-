"""
verification.py
---------------
Pre-flight integrity check for df_both.csv and the generated splits.

Run this script after generate_splits.py to verify that:
  1. df_both.csv has the required columns and no NaN in key fields.
  2. Class and modality distributions look reasonable.
  3. Every generated split CSV exists and has no subject leakage.
  4. Split size ratios are within expected bounds.

Usage:
    python verification.py \\
        --base_dir        /path/to/experiments \\
        --expname_unified exp_unified \\
        --expname_fdg     exp_fdg \\
        --expname_amy     exp_amy \\
        --n_iterations    25
"""

import os
import argparse
import pandas as pd


REQUIRED_COLUMNS = {"Filepath", "subject_id", "class_label", "Modality"}


def check_df_both(csv_path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Checking dataset: {csv_path}")
    print(f"{'='*60}")

    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    df = pd.read_csv(csv_path)

    # Required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    print(f"  ✓ Required columns present")

    # No NaN in key fields
    for col in REQUIRED_COLUMNS:
        n_nan = df[col].isna().sum()
        assert n_nan == 0, f"Column '{col}' has {n_nan} NaN values"
    print(f"  ✓ No NaN values in required columns")

    # Modality values
    mod_vals = set(df["Modality"].unique())
    assert mod_vals.issubset({0, 1}), f"Unexpected Modality values: {mod_vals}"
    print(f"  ✓ Modality values: {sorted(mod_vals)}")

    # Class label values
    cls_vals = set(df["class_label"].unique())
    assert cls_vals.issubset({0, 1}), f"Unexpected class_label values: {cls_vals}"
    print(f"  ✓ class_label values: {sorted(cls_vals)}")

    print(f"\n  Rows total : {len(df)}")
    print(f"  Subjects   : {df['subject_id'].nunique()}")
    print(f"\n  Class distribution:")
    print(df["class_label"].value_counts().rename({0: "CN (0)", 1: "AD (1)"}).to_string())
    print(f"\n  Modality distribution:")
    print(df["Modality"].value_counts().rename({0: "FDG (0)", 1: "AMY (1)"}).to_string())

    return df


def check_splits(split_dir: str, prefix: str, n_iterations: int,
                 df_filtered: pd.DataFrame,
                 expected_test_frac: float = 0.10,
                 expected_val_frac: float  = 0.15,
                 tolerance: float = 0.05):
    """
    For each split CSV:
      - Verify file exists.
      - Verify all three partitions are non-empty.
      - Verify no subject appears in more than one partition (no leakage).
      - Verify all subjects in the split appear in df_filtered.
      - Verify approximate split size ratios.
    """
    print(f"\n{'='*60}")
    print(f"  Checking splits: {split_dir}  (prefix={prefix})")
    print(f"{'='*60}")

    all_subjects = set(df_filtered["subject_id"].astype(str).unique())
    errors = []

    for i in range(1, n_iterations + 1):
        path = os.path.join(split_dir, f"split_{prefix}_iter{i:02d}.csv")
        if not os.path.exists(path):
            errors.append(f"  [Iter {i:02d}] Missing file: {path}")
            continue

        sdf = pd.read_csv(path)
        sdf["subject_id"] = sdf["subject_id"].astype(str)

        # Partition sizes
        counts = sdf["assigned_split"].value_counts()
        n_total = len(sdf)
        n_train = counts.get("train", 0)
        n_val   = counts.get("val",   0)
        n_test  = counts.get("test",  0)

        if n_train == 0 or n_val == 0 or n_test == 0:
            errors.append(f"  [Iter {i:02d}] Empty partition: "
                          f"train={n_train} val={n_val} test={n_test}")
            continue

        # No leakage
        train_ids = set(sdf.loc[sdf["assigned_split"]=="train", "subject_id"])
        val_ids   = set(sdf.loc[sdf["assigned_split"]=="val",   "subject_id"])
        test_ids  = set(sdf.loc[sdf["assigned_split"]=="test",  "subject_id"])

        overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        if overlap:
            errors.append(f"  [Iter {i:02d}] Subject leakage: {overlap}")

        # All subjects known
        unknown = set(sdf["subject_id"]) - all_subjects
        if unknown:
            errors.append(f"  [Iter {i:02d}] Unknown subjects: {unknown}")

        # Ratio check
        actual_test = n_test / n_total
        actual_val  = n_val  / n_total
        if abs(actual_test - expected_test_frac) > tolerance:
            errors.append(f"  [Iter {i:02d}] Test fraction {actual_test:.2f} "
                          f"deviates from expected {expected_test_frac:.2f} "
                          f"(tolerance={tolerance})")
        if abs(actual_val - expected_val_frac) > tolerance:
            errors.append(f"  [Iter {i:02d}] Val fraction {actual_val:.2f} "
                          f"deviates from expected {expected_val_frac:.2f} "
                          f"(tolerance={tolerance})")

    if errors:
        print("  ✗ Issues found:")
        for e in errors:
            print(e)
    else:
        print(f"  ✓ All {n_iterations} split files passed checks")


def main():
    parser = argparse.ArgumentParser(
        description="Verify df_both.csv and generated split CSVs."
    )
    parser.add_argument("--base_dir",        type=str, required=True)
    parser.add_argument("--expname_unified", type=str, default="exp_unified")
    parser.add_argument("--expname_fdg",     type=str, default="exp_fdg")
    parser.add_argument("--expname_amy",     type=str, default="exp_amy")
    parser.add_argument("--n_iterations",    type=int, default=25)
    parser.add_argument("--val_frac",        type=float, default=0.15)
    parser.add_argument("--test_frac",       type=float, default=0.10)
    args = parser.parse_args()

    csv_path = os.path.join(args.base_dir, args.expname_unified, "df_both.csv")
    df = check_df_both(csv_path)
    df["subject_id"] = df["subject_id"].astype(str)

    df_fdg = df[df["Modality"] == 0].copy()
    df_amy = df[df["Modality"] == 1].copy()

    configs = [
        ("unified", args.expname_unified, df),
        ("fdg",     args.expname_fdg,     df_fdg),
        ("amy",     args.expname_amy,     df_amy),
    ]

    for prefix, expname, df_filt in configs:
        split_dir = os.path.join(args.base_dir, expname, "splits")
        check_splits(
            split_dir=split_dir,
            prefix=prefix,
            n_iterations=args.n_iterations,
            df_filtered=df_filt,
            expected_test_frac=args.test_frac,
            expected_val_frac=args.val_frac,
        )

    print(f"\n✓ Verification complete.\n")


if __name__ == "__main__":
    main()
