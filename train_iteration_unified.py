# -*- coding: utf-8 -*-
"""
train_iteration_unified.py
--------------------------
Trains ONE iteration of the unified (both-modality) model.
Intended to be called by a SLURM array job with --iteration $SLURM_ARRAY_TASK_ID,
or run sequentially in a loop.

Architecture:
  SFCN backbone → logit-level fusion with a modality embedding
  (ModalityLogitFusion). The modality index (0=FDG, 1=AMY) is encoded
  as a learned embedding and concatenated with the backbone logits before
  a small MLP head produces the final class scores.

Hyperparameters (fixed from paper):
  batch_size   = 6
  lr           = 1e-4
  weight_decay = 5e-4
  epochs       = 200
  patience     = 8  (early stopping on validation loss)
  image size   = 128 × 128 × 128
  model        = sfcn
  embed_dim    = 8
  dropout      = 0.5

Usage:
  python train_iteration_unified.py \\
      --expname   exp_unified \\
      --base_dir  /path/to/experiments \\
      --iteration 1 \\
      --seed      42

Expected directory layout (created automatically if absent):
  <base_dir>/<expname>/
      df_both.csv          ← shared dataset CSV (required)
      splits/              ← produced by generate_splits.py
      models/              ← best checkpoints saved here
      results/             ← per-iteration CSVs saved here
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
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
PARAMS = {
    "batch_size":   6,
    "lr":           1e-4,
    "weight_decay": 5e-4,
    "epochs":       200,
    "patience":     8,
    "imagex":       128,
    "imagey":       128,
    "imagez":       128,
    "model_name":   "sfcn",
    "embed_dim":    8,
    "num_classes":  2,
    "dropout":      0.5,   # used both during training and when loading the best model
}


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class ModalityLogitFusion(nn.Module):
    """
    Wraps an image backbone and fuses its logits with a modality embedding.

    Forward inputs:
      x_img : (B, 1, X, Y, Z) — normalised 3-D PET volume
      x_mod : (B,)            — modality index (0=FDG, 1=AMY)

    Returns:
      logits : (B, num_classes)
    """

    def __init__(self, base_model: nn.Module, n_modalities: int = 2,
                 embed_dim: int = 8, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.base  = base_model
        self.embed = nn.Embedding(num_embeddings=n_modalities, embedding_dim=embed_dim)
        self.head  = nn.Sequential(
            nn.Linear(num_classes + embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_img: torch.Tensor, x_mod: torch.Tensor) -> torch.Tensor:
        logits_base = self.base(x_img)
        if logits_base.ndim > 2:
            logits_base = logits_base.view(logits_base.size(0), -1)
        emb   = self.embed(x_mod)
        fused = torch.cat([logits_base, emb], dim=1)
        return self.head(fused)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class ImageDatasetWithModality(Dataset):
    """PyTorch Dataset that returns (image_tensor, modality_index, label)."""

    def __init__(self, image_files, labels, modalities, transform):
        self.files     = list(image_files)
        self.labels    = np.asarray(labels)
        self.mods      = np.asarray(modalities)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.transform(self.files[idx])
        y   = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        m   = torch.tensor(int(self.mods[idx]),   dtype=torch.long)
        return img, m, y


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
        Resize((PARAMS["imagex"], PARAMS["imagey"], PARAMS["imagez"])),
    ])


def make_loader(imgs, labels, mods, transform,
                shuffle: bool, g: torch.Generator) -> DataLoader:
    """Build a deterministic DataLoader for the given split arrays."""
    ds = ImageDatasetWithModality(imgs, labels, mods, transform)
    return DataLoader(
        ds,
        batch_size=PARAMS["batch_size"],
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
        return {
            "X": df_all.loc[mask, "Filepath"].to_numpy(),
            "y": df_all.loc[mask, "class_label"].to_numpy().astype(int),
            "m": df_all.loc[mask, "Modality"].to_numpy().astype(int),
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
    for x, m, y in loader:
        x, m, y = x.to(device), m.long().to(device), y.to(device)
        logits = model(x, m)
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


def compute_metrics(y_true: np.ndarray, logits: np.ndarray,
                    sample_losses: np.ndarray = None,
                    positive_class: int = 1) -> Dict[str, Any]:
    """
    Compute accuracy, AUROC, sensitivity, specificity, and confusion matrix
    entries for an arbitrary subset of samples.

    Returns NaN for all metrics when the subset is empty or contains only
    one class (AUROC undefined).
    """
    if y_true.size == 0 or logits.size == 0:
        return {k: np.nan for k in
                ["loss", "acc", "auroc", "sensitivity", "specificity",
                 "n_samples", "tn", "fp", "fn", "tp"]}

    y_pred = np.argmax(logits, axis=1)
    acc    = accuracy_score(y_true, y_pred)
    auroc  = np.nan
    try:
        if len(np.unique(y_true)) == 2:
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, positive_class]
            auroc = roc_auc_score(y_true, probs)
    except Exception:
        pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    loss = float(np.mean(sample_losses)) if sample_losses is not None else np.nan

    return {
        "loss":        loss,
        "acc":         float(acc),
        "auroc":       float(auroc) if not np.isnan(auroc) else np.nan,
        "sensitivity": float(sens)  if not np.isnan(sens)  else np.nan,
        "specificity": float(spec)  if not np.isnan(spec)  else np.nan,
        "n_samples":   int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@torch.no_grad()
def evaluate_detailed(model: nn.Module, loader: DataLoader,
                      device: torch.device) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate the best model on the test set and return metrics for:
      - aggregate   : all samples
      - fdg         : FDG-PET samples only (Modality == 0)
      - amy         : Amyloid-PET samples only (Modality == 1)
    """
    model.eval()
    all_logits, all_labels, all_mods, all_losses = [], [], [], []
    for x, m, y in loader:
        x, m, y = x.to(device), m.long().to(device), y.to(device)
        logits = model(x, m)
        losses = F.cross_entropy(logits, y, reduction="none")
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_mods.append(m.detach().cpu())
        all_losses.append(losses.detach().cpu())

    y_true   = torch.cat(all_labels).numpy()
    logits   = torch.cat(all_logits).numpy()
    mods     = torch.cat(all_mods).numpy()
    s_losses = torch.cat(all_losses).numpy()

    agg = compute_metrics(y_true,          logits,          s_losses)
    fdg = compute_metrics(y_true[mods==0], logits[mods==0], s_losses[mods==0])
    amy = compute_metrics(y_true[mods==1], logits[mods==1], s_losses[mods==1])
    return {"aggregate": agg, "fdg": fdg, "amy": amy}


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_one_iteration(iteration: int, data: Dict,
                        device: torch.device, workdir: str,
                        iteration_seed: int) -> Dict[str, Any]:
    """
    Full training loop for one random-split iteration.

    Saves:
      models/best_model_unified_iter{N:02d}.pth
      results/history_unified_iter{N:02d}.csv

    Returns a dict of scalar metrics for this iteration.
    """
    set_seed(iteration_seed)
    g = torch.Generator()
    g.manual_seed(iteration_seed)

    tf           = make_transform()
    train_loader = make_loader(data["train"]["X"], data["train"]["y"],
                               data["train"]["m"], tf, True,  g)
    val_loader   = make_loader(data["val"]["X"],   data["val"]["y"],
                               data["val"]["m"],   tf, False, g)
    test_loader  = make_loader(data["test"]["X"],  data["test"]["y"],
                               data["test"]["m"],  tf, False, g)

    # Build model
    base  = get_model(PARAMS["model_name"], spatial_dims=3,
                      in_channels=1, out_channels=2).to(device)
    model = ModalityLogitFusion(
        base,
        n_modalities=2,
        embed_dim=PARAMS["embed_dim"],
        num_classes=PARAMS["num_classes"],
        dropout=PARAMS["dropout"],
    ).to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=PARAMS["lr"],
                                 weight_decay=PARAMS["weight_decay"])

    best_val_loss = float("inf")
    best_epoch    = -1
    no_improve    = 0
    history       = []
    ckpt_path     = os.path.join(workdir, "models",
                                 f"best_model_unified_iter{iteration:02d}.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, PARAMS["epochs"] + 1):
        model.train()
        train_losses = []
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.long().to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, m), y)
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
            f"[Unified | Iter {iteration:02d} | Epoch {epoch:03d}] "
            f"train={train_loss:.4f} val={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} auroc={val_metrics['auroc']:.4f} "
            f"best={best_val_loss:.4f}@{best_epoch} "
            f"no_imp={no_improve}/{PARAMS['patience']}"
        )

        if no_improve >= PARAMS["patience"]:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # ── Evaluate best checkpoint ───────────────────────────────────────────
    # Re-instantiate the architecture with the same hyperparameters used
    # during training, then load the saved state dict.
    base_best  = get_model(PARAMS["model_name"], spatial_dims=3,
                           in_channels=1, out_channels=2).to(device)
    best_model = ModalityLogitFusion(
        base_best,
        n_modalities=2,
        embed_dim=PARAMS["embed_dim"],
        num_classes=PARAMS["num_classes"],
        dropout=PARAMS["dropout"],   # same as training; Dropout is inactive in .eval()
    ).to(device)
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_eval    = evaluate_epoch(best_model, val_loader, device, nn.CrossEntropyLoss())
    test_detail = evaluate_detailed(best_model, test_loader, device)

    # Save epoch-level history
    pd.DataFrame(history).to_csv(
        os.path.join(workdir, "results", f"history_unified_iter{iteration:02d}.csv"),
        index=False,
    )

    agg = test_detail["aggregate"]
    fdg = test_detail["fdg"]
    amy = test_detail["amy"]

    return {
        "iteration":  iteration,
        "seed":       iteration_seed,
        "best_epoch": best_epoch,
        # Validation (best checkpoint)
        "val_loss":  val_eval["loss"],
        "val_acc":   val_eval["acc"],
        "val_auroc": val_eval["auroc"],
        # Test — aggregate
        "test_loss":         agg["loss"],
        "test_acc":          agg["acc"],
        "test_auroc":        agg["auroc"],
        "test_sensitivity":  agg["sensitivity"],
        "test_specificity":  agg["specificity"],
        "test_tn":           agg["tn"],
        "test_fp":           agg["fp"],
        "test_fn":           agg["fn"],
        "test_tp":           agg["tp"],
        # Test — FDG subset
        "test_fdg_acc":         fdg["acc"],
        "test_fdg_auroc":       fdg["auroc"],
        "test_fdg_sensitivity": fdg["sensitivity"],
        "test_fdg_specificity": fdg["specificity"],
        "test_fdg_n":           fdg["n_samples"],
        # Test — AMY subset
        "test_amy_acc":         amy["acc"],
        "test_amy_auroc":       amy["auroc"],
        "test_amy_sensitivity": amy["sensitivity"],
        "test_amy_specificity": amy["specificity"],
        "test_amy_n":           amy["n_samples"],
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
        description="Train one iteration of the unified FDG+AMY model."
    )
    parser.add_argument("--expname",   type=str, required=True,
                        help="Experiment folder name inside base_dir")
    parser.add_argument("--base_dir",  type=str, default="experiments",
                        help="Root directory containing all experiment folders")
    parser.add_argument("--iteration", type=int, required=True,
                        help="Iteration index (1-based). Pass $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Base seed; iteration seed = seed + iteration - 1")
    args = parser.parse_args()

    workdir = os.path.join(args.base_dir, args.expname)
    os.makedirs(os.path.join(workdir, "results"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "models"),  exist_ok=True)

    iteration      = args.iteration
    iteration_seed = args.seed + iteration - 1

    # Idempotent: skip if this iteration already finished successfully
    result_path = os.path.join(workdir, "results",
                               f"result_unified_iter{iteration:02d}.csv")
    if os.path.exists(result_path):
        print(f"[Iter {iteration:02d}] Already completed. Skipping.")
        return

    # Load shared dataset CSV
    df_all = pd.read_csv(os.path.join(workdir, "df_both.csv"))
    df_all["subject_id"]  = df_all["subject_id"].astype(str)
    df_all["class_label"] = df_all["class_label"].astype(int)
    df_all["Modality"]    = df_all["Modality"].astype(int)

    split_csv = os.path.join(workdir, "splits",
                             f"split_unified_iter{iteration:02d}.csv")
    assert os.path.exists(split_csv), \
        f"Split file not found: {split_csv}\nRun generate_splits.py first."

    data   = load_split_and_data(split_csv, df_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Unified | Iter {iteration:02d}] seed={iteration_seed} | device={device}")
    print(f"  train={len(data['train']['X'])} | "
          f"val={len(data['val']['X'])} | "
          f"test={len(data['test']['X'])}")

    summary = train_one_iteration(
        iteration=iteration,
        data=data,
        device=device,
        workdir=workdir,
        iteration_seed=iteration_seed,
    )

    pd.DataFrame([summary]).to_csv(result_path, index=False)
    print(f"\n[Iter {iteration:02d}] Saved result → {result_path}")
    print(f"  test_acc={summary['test_acc']:.4f} | "
          f"test_auroc={summary['test_auroc']:.4f} | "
          f"test_sens={summary['test_sensitivity']:.4f} | "
          f"test_spec={summary['test_specificity']:.4f}")


if __name__ == "__main__":
    main()
