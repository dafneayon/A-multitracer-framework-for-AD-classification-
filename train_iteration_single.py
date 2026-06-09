# -*- coding: utf-8 -*-
"""
train_iteration_single.py
--------------------------
Trains ONE iteration of a single-modality (unimodal) baseline model.
Supports FDG-PET or Amyloid-PET (AMY) independently.
Intended to be called by a SLURM array job with --iteration $SLURM_ARRAY_TASK_ID,
or run sequentially in a loop.

Architecture:
  Plain SFCN classifier (no modality embedding). Each baseline model sees only
  scans from its own tracer.

Hyperparameters per modality (fixed from paper):
  FDG : batch_size=4, lr=3.34e-4, weight_decay=8.46e-4, epochs=60, patience=8
  AMY : batch_size=6, lr=2.30e-4, weight_decay=2.72e-6, epochs=60, patience=8
  Both: image size = 128 × 128 × 128, model = sfcn

Usage:
  python train_iteration_single.py \\
      --expname          exp_fdg \\
      --base_dir         /path/to/experiments \\
      --target_modality  fdg \\
      --iteration        1 \\
      --seed             42

Expected directory layout (created automatically if absent):
  <base_dir>/<expname>/
      df_both.csv       ← shared dataset CSV (required; FDG/AMY rows filtered here)
      splits/           ← produced by generate_splits.py
      models/           ← best checkpoints saved here
      results/          ← per-iteration CSVs saved here
"""

import os
import sys
import argparse
import logging
import random
from typing import Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize, LoadImage

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from utils import get_model


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters per modality
# ──────────────────────────────────────────────────────────────────────────────
PARAMS = {
    "fdg": {
        "batch_size":    4,
        "lr":            3.34e-4,
        "weight_decay":  8.46e-4,
        "epochs":        60,
        "patience":      8,
        "modality_code": 0,
    },
    "amy": {
        "batch_size":    6,
        "lr":            2.30e-4,
        "weight_decay":  2.72e-6,
        "epochs":        60,
        "patience":      8,
        "modality_code": 1,
    },
}

# Shared across both modalities
SHARED = {
    "imagex":      128,
    "imagey":      128,
    "imagez":      128,
    "model_name":  "sfcn",
    "num_classes": 2,
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class ImageDatasetSingleModality(Dataset):
    """PyTorch Dataset that returns (image_tensor, label) for a single-tracer split."""

    def __init__(self, image_files, labels, transform):
        self.files     = list(image_files)
        self.labels    = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.transform(self.files[idx])
        y   = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, y


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_transform() -> Compose:
    """Standard MONAI preprocessing pipeline: load → intensity scale → channel → resize."""
    return Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((SHARED["imagex"], SHARED["imagey"], SHARED["imagez"])),
    ])


def make_loader(imgs, labels, transform, shuffle: bool,
                g: torch.Generator, batch_size: int) -> DataLoader:
    """Build a deterministic DataLoader for the given split arrays."""
    ds = ImageDatasetSingleModality(imgs, labels, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % 2**32),
        generator=g,
        pin_memory=False,
    )


def load_split_and_data(split_csv: str,
                        df_all: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Read the pre-generated split CSV and return train/val/test arrays.

    Asserts that no subject appears in more than one partition (leakage check).
    """
    split_df = pd.read_csv(split_csv)
    split_df["subject_id"] = split_df["subject_id"].astype(str)
    df_all["subject_id"]   = df_all["subject_id"].astype(str)

    def _mask(partition):
        sids = set(split_df.loc[split_df["assigned_split"] == partition, "subject_id"])
        return df_all["subject_id"].isin(sids)

    tr, va, te = _mask("train"), _mask("val"), _mask("test")

    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any(), \
        "Subject leakage detected across splits!"

    def _arrays(mask):
        sub = df_all.loc[mask]
        return {
            "X":    sub["Filepath"].to_numpy(),
            "y":    sub["class_label"].to_numpy().astype(int),
            "meta": sub[["Filepath", "subject_id", "class_label"]].reset_index(drop=True),
        }

    return {"train": _arrays(tr), "val": _arrays(va), "test": _arrays(te)}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader,
                   device: torch.device, loss_fn) -> Dict[str, Any]:
    """Compute loss, accuracy, and AUROC over an entire DataLoader."""
    model.eval()
    losses, all_logits, all_labels = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    y_true = torch.cat(all_labels).numpy()
    logits  = torch.cat(all_logits).numpy()
    y_pred  = np.argmax(logits, axis=1)
    acc     = accuracy_score(y_true, y_pred)
    auroc   = np.nan
    try:
        if len(np.unique(y_true)) == 2:
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            auroc = roc_auc_score(y_true, probs)
    except Exception:
        pass
    return {
        "loss":  float(np.mean(losses)),
        "acc":   float(acc),
        "auroc": float(auroc) if not np.isnan(auroc) else np.nan,
    }


@torch.no_grad()
def evaluate_detailed(model: nn.Module, loader: DataLoader,
                      device: torch.device,
                      positive_class: int = 1) -> Dict[str, Any]:
    """
    Compute full classification metrics including sensitivity, specificity,
    per-sample predictions, and probabilities for the test set.
    """
    model.eval()
    all_logits, all_labels, all_losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_losses.append(losses.detach().cpu())

    y_true   = torch.cat(all_labels).numpy()
    logits   = torch.cat(all_logits).numpy()
    s_losses = torch.cat(all_losses).numpy()
    y_pred   = np.argmax(logits, axis=1)
    acc      = accuracy_score(y_true, y_pred)

    auroc     = np.nan
    probs_pos = np.full(len(y_true), np.nan)
    try:
        if len(np.unique(y_true)) == 2:
            probs_pos = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, positive_class]
            auroc = roc_auc_score(y_true, probs_pos)
    except Exception:
        pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "loss":        float(np.mean(s_losses)),
        "acc":         float(acc),
        "auroc":       float(auroc) if not np.isnan(auroc) else np.nan,
        "sensitivity": float(sens)  if not np.isnan(sens)  else np.nan,
        "specificity": float(spec)  if not np.isnan(spec)  else np.nan,
        "n_samples":   int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "probs_pos":   probs_pos,
        "preds":       y_pred,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_one_iteration(iteration: int, data: Dict,
                        device: torch.device, workdir: str,
                        iteration_seed: int,
                        target_modality: str) -> Dict[str, Any]:
    """
    Full training loop for one random-split iteration of a single-tracer model.

    Saves:
      models/best_model_{modality}_iter{N:02d}.pth
      results/history_{modality}_iter{N:02d}.csv
      results/test_preds_{modality}_iter{N:02d}.csv

    Returns a dict of scalar metrics for this iteration.
    """
    p = {**PARAMS[target_modality], **SHARED}   # merged params for this modality

    set_seed(iteration_seed)
    g = torch.Generator()
    g.manual_seed(iteration_seed)

    tf           = make_transform()
    train_loader = make_loader(data["train"]["X"], data["train"]["y"],
                               tf, True,  g, p["batch_size"])
    val_loader   = make_loader(data["val"]["X"],   data["val"]["y"],
                               tf, False, g, p["batch_size"])
    test_loader  = make_loader(data["test"]["X"],  data["test"]["y"],
                               tf, False, g, p["batch_size"])

    model = get_model(p["model_name"], spatial_dims=3,
                      in_channels=1, out_channels=p["num_classes"]).to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=p["lr"], weight_decay=p["weight_decay"])

    best_val_loss = float("inf")
    best_epoch    = -1
    no_improve    = 0
    history       = []
    ckpt_path = os.path.join(workdir, "models",
                             f"best_model_{target_modality}_iter{iteration:02d}.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, p["epochs"] + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss  = float(np.mean(train_losses))
        val_metrics = evaluate_epoch(model, val_loader, device, loss_fn)

        history.append({
            "iteration":  iteration,
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["acc"],
            "val_auroc":  val_metrics["auroc"],
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        print(
            f"[{target_modality.upper()} | Iter {iteration:02d} | Epoch {epoch:03d}] "
            f"train={train_loss:.4f} val={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} auroc={val_metrics['auroc']:.4f} "
            f"best={best_val_loss:.4f}@{best_epoch} "
            f"no_imp={no_improve}/{p['patience']}"
        )

        if no_improve >= p["patience"]:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # ── Evaluate best checkpoint ───────────────────────────────────────────
    best_model = get_model(p["model_name"], spatial_dims=3,
                           in_channels=1, out_channels=p["num_classes"]).to(device)
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_eval    = evaluate_epoch(best_model, val_loader, device, nn.CrossEntropyLoss())
    test_detail = evaluate_detailed(best_model, test_loader, device, positive_class=1)

    # Save epoch-level history
    os.makedirs(os.path.join(workdir, "results"), exist_ok=True)
    pd.DataFrame(history).to_csv(
        os.path.join(workdir, "results",
                     f"history_{target_modality}_iter{iteration:02d}.csv"),
        index=False,
    )

    # Save per-sample test predictions
    test_meta = data["test"]["meta"].copy()
    test_meta["preds"]   = test_detail["preds"]
    test_meta["prob_ad"] = test_detail["probs_pos"]
    test_meta["TP"] = ((test_meta["class_label"]==1) & (test_meta["preds"]==1)).astype(int)
    test_meta["TN"] = ((test_meta["class_label"]==0) & (test_meta["preds"]==0)).astype(int)
    test_meta["FP"] = ((test_meta["class_label"]==0) & (test_meta["preds"]==1)).astype(int)
    test_meta["FN"] = ((test_meta["class_label"]==1) & (test_meta["preds"]==0)).astype(int)
    test_meta.to_csv(
        os.path.join(workdir, "results",
                     f"test_preds_{target_modality}_iter{iteration:02d}.csv"),
        index=False,
    )

    return {
        "iteration":  iteration,
        "seed":       iteration_seed,
        "best_epoch": best_epoch,
        # Validation (best checkpoint)
        "val_loss":  val_eval["loss"],
        "val_acc":   val_eval["acc"],
        "val_auroc": val_eval["auroc"],
        # Test
        "test_loss":        test_detail["loss"],
        "test_acc":         test_detail["acc"],
        "test_auroc":       test_detail["auroc"],
        "test_sensitivity": test_detail["sensitivity"],
        "test_specificity": test_detail["specificity"],
        "test_tn":          test_detail["tn"],
        "test_fp":          test_detail["fp"],
        "test_fn":          test_detail["fn"],
        "test_tp":          test_detail["tp"],
        # Split sizes
        "n_train": len(data["train"]["X"]),
        "n_val":   len(data["val"]["X"]),
        "n_test":  len(data["test"]["X"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Train one iteration of a single-tracer (FDG or AMY) baseline model."
    )
    parser.add_argument("--expname",          type=str, required=True,
                        help="Experiment folder name inside base_dir")
    parser.add_argument("--base_dir",         type=str, default="experiments",
                        help="Root directory containing all experiment folders")
    parser.add_argument("--target_modality",  type=str, required=True,
                        choices=["fdg", "amy"],
                        help="Which tracer to train on")
    parser.add_argument("--iteration",        type=int, required=True,
                        help="Iteration index (1-based). Pass $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--seed",             type=int, default=42,
                        help="Base seed; iteration seed = seed + iteration - 1")
    args = parser.parse_args()

    workdir = os.path.join(args.base_dir, args.expname)
    os.makedirs(os.path.join(workdir, "results"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "models"),  exist_ok=True)

    iteration      = args.iteration
    iteration_seed = args.seed + iteration - 1

    # Idempotent: skip if this iteration already finished successfully
    result_path = os.path.join(
        workdir, "results",
        f"result_{args.target_modality}_iter{iteration:02d}.csv"
    )
    if os.path.exists(result_path):
        print(f"[{args.target_modality.upper()} | Iter {iteration:02d}] "
              f"Already completed. Skipping.")
        return

    # Load and filter dataset to the target modality
    df_all = pd.read_csv(os.path.join(workdir, "df_both.csv"))
    df_all["subject_id"]  = df_all["subject_id"].astype(str)
    df_all["class_label"] = df_all["class_label"].astype(int)
    df_all["Modality"]    = df_all["Modality"].astype(int)

    mod_code = PARAMS[args.target_modality]["modality_code"]
    df_all   = df_all[df_all["Modality"] == mod_code].copy().reset_index(drop=True)

    split_csv = os.path.join(
        workdir, "splits",
        f"split_{args.target_modality}_iter{iteration:02d}.csv"
    )
    assert os.path.exists(split_csv), \
        f"Split file not found: {split_csv}\nRun generate_splits.py first."

    data   = load_split_and_data(split_csv, df_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[{args.target_modality.upper()} | Iter {iteration:02d}] "
          f"seed={iteration_seed} | device={device}")
    print(f"  train={len(data['train']['X'])} | "
          f"val={len(data['val']['X'])} | "
          f"test={len(data['test']['X'])}")
    print(f"  lr={PARAMS[args.target_modality]['lr']} | "
          f"batch={PARAMS[args.target_modality]['batch_size']} | "
          f"wd={PARAMS[args.target_modality]['weight_decay']}")

    summary = train_one_iteration(
        iteration=iteration,
        data=data,
        device=device,
        workdir=workdir,
        iteration_seed=iteration_seed,
        target_modality=args.target_modality,
    )

    pd.DataFrame([summary]).to_csv(result_path, index=False)
    print(f"\n[Iter {iteration:02d}] Saved result → {result_path}")
    print(f"  test_acc={summary['test_acc']:.4f} | "
          f"test_auroc={summary['test_auroc']:.4f} | "
          f"test_sens={summary['test_sensitivity']:.4f} | "
          f"test_spec={summary['test_specificity']:.4f}")


if __name__ == "__main__":
    main()
