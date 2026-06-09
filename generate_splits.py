"""
generate_splits.py
------------------
Runs ONCE on CPU before launching any SLURM array.
Reads df_both.csv from a single shared location (exp_unified folder)
since it is the same dataset for all 3 models.

Generates 25 stratified subject-level splits for each model:
  - unified  (both modalities, strata = diagnosis x modality profile)
  - fdg      (FDG only,        strata = diagnosis)
  - amy      (AMY only,        strata = diagnosis)

Output structure:
  exp_unified/splits/split_unified_iter01.csv  ...  split_unified_iter25.csv
  exp_fdg/splits/split_fdg_iter01.csv          ...  split_fdg_iter25.csv
  exp_amy/splits/split_amy_iter01.csv          ...  split_amy_iter25.csv

Usage:
  python generate_splits.py \
      --base_dir /path/to/experiments \
      --expname_unified exp_unified \
      --expname_fdg     exp_fdg \
      --expname_amy     exp_amy \
      --seed 42 \
      --n_iterations 25
"""

import os
import argparse
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


# ──────────────────────────────────────────────
# Subject table builders
# ──────────────────────────────────────────────
def build_subject_table_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per subject.
    Strata = diagnosis (CN/AD) x modality profile (fdg_only / amy_only / both).
    """
    rows = []
    for subject_id, sdf in df.groupby("subject_id"):
        class_values = sdf["class_label"].dropna().unique()
        if len(class_values) != 1:
            raise ValueError(
                f"Subject {subject_id} has inconsistent class_label: {class_values}"
            )
        y       = int(class_values[0])
        has_fdg = int((sdf["Modality"] == 0).any())
        has_amy = int((sdf["Modality"] == 1).any())

        if has_fdg and has_amy:
            mod_profile = "both"
        elif has_fdg:
            mod_profile = "fdg_only"
        elif has_amy:
            mod_profile = "amy_only"
        else:
            raise ValueError(f"Subject {subject_id} has no valid modality rows.")

        dx_name = "ad" if y == 1 else "cn"
        rows.append({
            "subject_id":    str(subject_id),
            "class_label":   y,
            "has_fdg":       has_fdg,
            "has_amy":       has_amy,
            "mod_profile":   mod_profile,
            "split_stratum": f"{dx_name}_{mod_profile}",
        })
    return pd.DataFrame(rows)


def build_subject_table_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per subject (already filtered to one modality).
    Strata = diagnosis only.
    """
    rows = []
    for subject_id, sdf in df.groupby("subject_id"):
        class_values = sdf["class_label"].dropna().unique()
        if len(class_values) != 1:
            raise ValueError(
                f"Subject {subject_id} has inconsistent class_label: {class_values}"
            )
        y = int(class_values[0])
        rows.append({
            "subject_id":    str(subject_id),
            "class_label":   y,
            "split_stratum": y,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Split logic
# ──────────────────────────────────────────────
def _two_stage_split(subject_df: pd.DataFrame,
                     seed: int,
                     test_frac: float,
                     val_frac: float,
                     stratify_col: Optional[str]
                     ) -> Tuple[List[str], List[str], List[str]]:
    ids   = subject_df["subject_id"].to_numpy()
    strat = subject_df[stratify_col].to_numpy() if stratify_col else None

    trainval, test = train_test_split(
        ids, test_size=test_frac, random_state=seed, stratify=strat
    )
    remain       = subject_df[subject_df["subject_id"].isin(trainval)].copy()
    remain_strat = remain[stratify_col].to_numpy() if stratify_col else None

    train, val = train_test_split(
        remain["subject_id"].to_numpy(),
        test_size=val_frac / (1.0 - test_frac),
        random_state=seed + 1,
        stratify=remain_strat
    )
    return list(train), list(val), list(test)


def split_subjects(subject_df: pd.DataFrame,
                   seed: int,
                   test_frac: float = 0.10,
                   val_frac: float  = 0.15,
                   strategies: Optional[List[Optional[str]]] = None
                   ) -> Tuple[List[str], List[str], List[str], str]:
    """
    Try stratification strategies in order, fall back gracefully.
    Returns (train_ids, val_ids, test_ids, strategy_used).
    """
    if strategies is None:
        strategies = ["split_stratum", "class_label", None]

    last_err = None
    for col in strategies:
        try:
            tr, va, te = _two_stage_split(
                subject_df, seed, test_frac, val_frac, col
            )
            used = col if col is not None else "no_stratification"
            return tr, va, te, used
        except ValueError as e:
            last_err = e

    raise RuntimeError(f"Could not split subjects. Last error: {last_err}")


def save_split_csv(subject_df: pd.DataFrame,
                   train_subjects: List[str],
                   val_subjects:   List[str],
                   test_subjects:  List[str],
                   out_path: str,
                   seed: int,
                   strat_used: str):
    split_map = (
        {s: "train" for s in train_subjects} |
        {s: "val"   for s in val_subjects}   |
        {s: "test"  for s in test_subjects}
    )
    out = subject_df.copy()
    out["assigned_split"]      = out["subject_id"].map(split_map)
    out["seed"]                = seed
    out["stratification_used"] = strat_used
    out.to_csv(out_path, index=False)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",        type=str, required=True,
                        help="Root directory containing all experiment folders "
                             "(e.g. /path/to/experiments)")
    parser.add_argument("--expname_unified", type=str, default="exp_unified")
    parser.add_argument("--expname_fdg",     type=str, default="exp_fdg")
    parser.add_argument("--expname_amy",     type=str, default="exp_amy")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--n_iterations",    type=int, default=25)
    parser.add_argument("--val_frac",        type=float, default=0.15)
    parser.add_argument("--test_frac",       type=float, default=0.10)
    args = parser.parse_args()

    set_seed(args.seed)

    # ── df_both.csv is SHARED → read once from unified dir ────
    csv_path = os.path.join(args.base_dir, args.expname_unified, "df_both.csv")
    assert os.path.exists(csv_path), \
        f"df_both.csv not found at: {csv_path}\n" \
        f"Place the shared dataset CSV in the unified experiment folder."

    # ── create split output dirs ───────────────────────────────
    split_dirs = {}
    for key, expname in [("unified", args.expname_unified),
                         ("fdg",     args.expname_fdg),
                         ("amy",     args.expname_amy)]:
        sd = os.path.join(args.base_dir, expname, "splits")
        os.makedirs(sd, exist_ok=True)
        split_dirs[key] = sd

    # ── load & cast ────────────────────────────────────────────
    df_all = pd.read_csv(csv_path)
    required = {"Filepath", "class_label", "Modality", "subject_id"}
    assert required.issubset(df_all.columns), \
        f"Missing columns: {required - set(df_all.columns)}"

    df_all["subject_id"]  = df_all["subject_id"].astype(str)
    df_all["class_label"] = df_all["class_label"].astype(int)
    df_all["Modality"]    = df_all["Modality"].astype(int)

    # ── filter per modality ────────────────────────────────────
    df_fdg = df_all[df_all["Modality"] == 0].copy().reset_index(drop=True)
    df_amy = df_all[df_all["Modality"] == 1].copy().reset_index(drop=True)

    # ── subject tables ─────────────────────────────────────────
    subj_unified = build_subject_table_unified(df_all)
    subj_fdg     = build_subject_table_single(df_fdg)
    subj_amy     = build_subject_table_single(df_amy)

    # ── print dataset summary ──────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Dataset: {csv_path}")
    print(f"{'='*55}")
    print(f"  Total rows       : {len(df_all)}")
    print(f"  FDG rows         : {len(df_fdg)}")
    print(f"  AMY rows         : {len(df_amy)}")
    print(f"  Subjects unified : {len(subj_unified)}")
    print(f"  Subjects FDG     : {len(subj_fdg)}")
    print(f"  Subjects AMY     : {len(subj_amy)}")
    print("\n  Unified strata (diagnosis x modality profile):")
    print(subj_unified["split_stratum"].value_counts().sort_index().to_string())
    print("\n  FDG class distribution (CN=0, AD=1):")
    print(subj_fdg["class_label"].value_counts().sort_index().to_string())
    print("\n  AMY class distribution (CN=0, AD=1):")
    print(subj_amy["class_label"].value_counts().sort_index().to_string())

    # ── generate & save all splits ─────────────────────────────
    print(f"\nGenerating {args.n_iterations} splits "
          f"(seed base={args.seed}, "
          f"train/val/test = "
          f"{1-args.val_frac-args.test_frac:.0%}/"
          f"{args.val_frac:.0%}/"
          f"{args.test_frac:.0%}) …\n")

    for iteration in range(1, args.n_iterations + 1):
        iter_seed = args.seed + iteration - 1

        # Unified — richest stratification
        tr, va, te, strat = split_subjects(
            subj_unified, iter_seed, args.test_frac, args.val_frac,
            strategies=["split_stratum", "class_label", None]
        )
        save_split_csv(
            subj_unified, tr, va, te,
            out_path=os.path.join(
                split_dirs["unified"],
                f"split_unified_iter{iteration:02d}.csv"
            ),
            seed=iter_seed, strat_used=strat
        )

        # FDG
        tr, va, te, strat = split_subjects(
            subj_fdg, iter_seed, args.test_frac, args.val_frac,
            strategies=["split_stratum", None]
        )
        save_split_csv(
            subj_fdg, tr, va, te,
            out_path=os.path.join(
                split_dirs["fdg"],
                f"split_fdg_iter{iteration:02d}.csv"
            ),
            seed=iter_seed, strat_used=strat
        )

        # AMY
        tr, va, te, strat = split_subjects(
            subj_amy, iter_seed, args.test_frac, args.val_frac,
            strategies=["split_stratum", None]
        )
        save_split_csv(
            subj_amy, tr, va, te,
            out_path=os.path.join(
                split_dirs["amy"],
                f"split_amy_iter{iteration:02d}.csv"
            ),
            seed=iter_seed, strat_used=strat
        )

        print(f"  [Iter {iteration:02d}] seed={iter_seed} ✓")

    print(f"\n All {args.n_iterations} splits generated for 3 models.")
    for key, sd in split_dirs.items():
        print(f"  {key:8s}: {sd}")


if __name__ == "__main__":
    main()