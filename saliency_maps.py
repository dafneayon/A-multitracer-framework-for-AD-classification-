#!/usr/bin/env python3
"""
saliency_maps.py
----------------
Generates SmoothGrad saliency maps for PET images listed in a CSV file,
using the trained unified SFCN model (ModalityFeatureFusion architecture).

For each subject, the script:
  1. Loads and preprocesses the PET volume (intensity scaling, channel first).
  2. Runs a forward pass to obtain the predicted class.
  3. Computes SmoothGrad attributions via Captum's NoiseTunnel + Saliency.
  4. Saves the saliency map as a NIfTI file (.nii.gz) with the correct affine
     matrix reconstructed from the original image header.
  5. Optionally saves PNG overlay figures (axial, coronal, sagittal) on the
     MNI152 template using nilearn.

The script handles CUDA out-of-memory errors automatically by retrying on CPU.

Architecture note:
  This script is designed for the unified model only (ModalityFeatureFusion),
  which takes both an image tensor and a modality index as inputs.
  The SFCN backbone runs in feature-extraction mode (SFCNEncoder), and its
  64-dimensional embedding is fused with a learned modality embedding before
  the classification head.

Usage:
    python saliency_maps.py \
        --exp_name   exp_unified \
        --model_name sfcn \
        --csv        /path/to/subjects.csv \
        --save_dir   /path/to/output \
        --nsamples   20 \
        --sigma      0.15 \
        --device     cuda

    # Point to your experiments root directory:
    export EXPERIMENT_BASE_DIR=/path/to/experiments

Input CSV columns required:
    Filepath     : absolute path to the preprocessed NIfTI image
    Modality     : 0 = FDG-PET, 1 = Amyloid-PET
    class_label  : 0 = CN, 1 = AD

Output (per subject, saved to --save_dir):
    saliency_<name>.nii.gz   : saliency map in original image space
    <name>_axial.png         : axial overlay on MNI152 template
    <name>_coronal.png       : coronal overlay
    <name>_sagittal.png      : sagittal overlay
"""

import os
import sys
import gc
import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch
import monai
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureChannelFirstd,
)

try:
    import config_file as cfg           # default: config_file.py in this repo
except ImportError:
    import config_file as cfg     # legacy alias for backward compatibility

from utils import get_model


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────
class ModalityFeatureFusion(torch.nn.Module):
    """
    Unified model that fuses image features with a modality embedding.

    The SFCN encoder produces a (B, image_feat_dim) feature vector.
    A learned embedding maps the modality index to (B, embed_dim).
    Both are concatenated and passed through a small MLP head.

    This architecture mirrors the one used in train_iteration_unified.py
    but operates at the feature level (using SFCNEncoder) rather than at
    the logit level (ModalityLogitFusion), making it compatible with
    gradient-based attribution methods.

    Args:
        image_encoder  : Pretrained SFCNEncoder instance.
        image_feat_dim : Dimensionality of the encoder output (default: 64).
        n_modalities   : Number of modalities (default: 2).
        embed_dim      : Dimensionality of the modality embedding (default: 8).
        num_classes    : Number of output classes (default: 2).
        dropout        : Dropout probability in the MLP head (default: 0.5).
    """

    def __init__(
        self,
        image_encoder: torch.nn.Module,
        image_feat_dim: int = 64,
        n_modalities: int = 2,
        embed_dim: int = 8,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = image_encoder
        self.embed   = torch.nn.Embedding(num_embeddings=n_modalities,
                                          embedding_dim=embed_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(image_feat_dim + embed_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x_img: torch.Tensor,
                x_mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_img : (B, 1, X, Y, Z) normalised PET volume.
            x_mod : (B,) modality index (0=FDG, 1=AMY).

        Returns:
            logits : (B, num_classes)
        """
        img_feat = self.encoder(x_img)
        mod_feat = self.embed(x_mod)
        return self.head(torch.cat([img_feat, mod_feat], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 1):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(exp_name: str, model_name: str,
                device: torch.device) -> ModalityFeatureFusion:
    """
    Instantiate the unified model and load the best saved checkpoint.

    Checkpoint search order:
      1. <EXPERIMENT_BASE_DIR>/<exp_name>/best_model_<exp_name>.pth
      2. <EXPERIMENT_BASE_DIR>/<exp_name>/<model_name>_last_model.pth

    Args:
        exp_name   : Experiment folder name (must match the training run).
        model_name : Architecture name ('sfcn' only).
        device     : Target device.

    Returns:
        Loaded ModalityFeatureFusion model in eval mode.

    Raises:
        ValueError        : If model_name is not 'sfcn'.
        FileNotFoundError : If no checkpoint is found.
    """
    if model_name.lower() != "sfcn":
        raise ValueError(
            f"This script supports the SFCN feature fusion architecture only. "
            f"Got: {model_name}"
        )

    image_encoder = get_model(
        model_name=model_name,
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        norm_type="instance",
        dropout=0.5,
        feature_mode=True,
        feature_dim=64,
    ).to(device)

    model = ModalityFeatureFusion(
        image_encoder=image_encoder,
        image_feat_dim=64,
        n_modalities=2,
        embed_dim=8,
        num_classes=2,
        dropout=0.5,
    ).to(device)

    base_dir    = os.environ.get("EXPERIMENT_BASE_DIR", "experiments")
    working_dir = os.path.join(base_dir, exp_name)
    cand_best   = os.path.join(working_dir, f"best_model_{exp_name}.pth")
    cand_last   = os.path.join(working_dir, f"{model_name}_last_model.pth")

    weight_path = cand_best if os.path.exists(cand_best) else cand_last
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"No checkpoint found at:\n  - {cand_best}\n  - {cand_last}\n"
            f"Set EXPERIMENT_BASE_DIR to your experiments root directory."
        )

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def make_transform() -> Compose:
    """
    MONAI preprocessing pipeline (matching training):
    load → scale intensity to [0,1] → ensure channel-first format.

    Spatial resizing is intentionally omitted so the saliency map
    can be saved with the original image affine.
    """
    return Compose([
        LoadImaged(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
    ])


def compute_resized_affine(orig_affine: np.ndarray,
                           orig_shape: tuple,
                           new_shape: tuple) -> np.ndarray:
    """
    Reconstruct the NIfTI affine for a spatially resized image.

    Preserves the original orientation and origin while scaling the
    voxel size to account for the change in spatial dimensions.

    Args:
        orig_affine : (4, 4) affine from the original NIfTI header.
        orig_shape  : Original spatial dimensions (X, Y, Z).
        new_shape   : New spatial dimensions after resizing (X', Y', Z').

    Returns:
        new_affine : (4, 4) adjusted affine matrix.
    """
    ox, oy, oz = orig_shape[:3]
    nx, ny, nz = new_shape[:3]
    A          = orig_affine.copy()
    A[:3, 0]   = orig_affine[:3, 0] * (ox / nx)
    A[:3, 1]   = orig_affine[:3, 1] * (oy / ny)
    A[:3, 2]   = orig_affine[:3, 2] * (oz / nz)
    return A


# ──────────────────────────────────────────────────────────────────────────────
# Overlay visualisation
# ──────────────────────────────────────────────────────────────────────────────
def save_overlay_pngs(
    t1_img,
    saliency_rs_img,
    out_prefix: str,
    outdir: str,
    alpha: float = 0.6,
    vmax_p: float = 99.0,
    axial_cuts: tuple = (-10, 0, 10, 20, 30, 40, 50),
    n_cuts_other: int = 7,
):
    """
    Save axial, coronal, and sagittal PNG overlays of a saliency map
    on a structural template using nilearn.

    The colour scale maximum (vmax) is set to the vmax_p-th percentile
    of non-zero absolute saliency values within the brain mask, preventing
    a small number of high-intensity voxels from washing out the colour map.

    Args:
        t1_img          : nilearn-compatible template (e.g. MNI152).
        saliency_rs_img : Saliency map resampled to template space.
        out_prefix      : Filename prefix for output PNGs.
        outdir          : Output directory.
        alpha           : Overlay transparency (default: 0.6).
        vmax_p          : Percentile for colour scale maximum (default: 99).
        axial_cuts      : Z-coordinates (mm) for axial slices.
        n_cuts_other    : Number of slices for coronal and sagittal views.
    """
    from nilearn.plotting import plot_anat
    from nilearn.masking import compute_brain_mask

    os.makedirs(os.path.abspath(outdir), exist_ok=True)

    mask  = compute_brain_mask(t1_img).get_fdata().astype(bool)
    s     = np.nan_to_num(saliency_rs_img.get_fdata().astype(np.float32))
    sb_nz = np.abs(s[mask])
    sb_nz = sb_nz[sb_nz != 0]
    vmax  = (float(np.percentile(sb_nz, vmax_p)) if sb_nz.size > 0
             else float(np.max(np.abs(s)) + 1e-6))

    for mode, cuts, label in [
        ("z", list(axial_cuts), "axial"),
        ("y", n_cuts_other,     "coronal"),
        ("x", n_cuts_other,     "sagittal"),
    ]:
        disp = plot_anat(
            t1_img, display_mode=mode, cut_coords=cuts,
            black_bg=True, annotate=True,
            title=f"{out_prefix} ({label})",
        )
        disp.add_overlay(saliency_rs_img, cmap="hot", alpha=alpha, vmax=vmax)
        disp.savefig(os.path.join(outdir, f"{out_prefix}_{label}.png"), dpi=200)
        disp.close()


# ──────────────────────────────────────────────────────────────────────────────
# Core attribution function
# ──────────────────────────────────────────────────────────────────────────────
def explain_one_case(
    model: torch.nn.Module,
    device: torch.device,
    image_path: str,
    modality: int,
    true_class: int,
    save_dir: str,
    nsamples: int,
    sigma: float,
    resize_xyz: tuple,
    nt_batch: int = 1,
    name_suffix: str = "",
    target_mode: str = "pred",
    save_overlays: bool = True,
):
    """
    Compute and save the SmoothGrad saliency map for a single PET image.

    SmoothGrad adds Gaussian noise to the input nsamples times and averages
    the resulting gradients, producing a smoother attribution map than
    vanilla gradient saliency. The absolute value is then normalised to [0,1].

    Args:
        model        : Loaded ModalityFeatureFusion in eval mode.
        device       : Inference device.
        image_path   : Path to the preprocessed NIfTI image.
        modality     : Modality index (0=FDG, 1=AMY).
        true_class   : Ground-truth label (0=CN, 1=AD).
        save_dir     : Output directory.
        nsamples     : Number of noisy samples for SmoothGrad.
        sigma        : Gaussian noise standard deviation.
        resize_xyz   : Training spatial size tuple; used only for affine
                       reconstruction (no resizing is applied here).
        nt_batch     : Forward-pass batch size inside NoiseTunnel.
        name_suffix  : Prefix appended to output filenames.
        target_mode  : 'pred' = attribute toward predicted class;
                       'true' = attribute toward ground-truth class.
        save_overlays: If True, also save PNG overlay figures.

    Returns:
        nii_out      : Path to the saved NIfTI saliency file.
        pred_class   : Model-predicted class index.
        target_class : Class index used for attribution.
    """
    from captum.attr import Saliency, NoiseTunnel
    from nilearn import datasets
    from nilearn.image import resample_to_img

    os.makedirs(save_dir, exist_ok=True)

    # Preprocess image
    data  = make_transform()({"img": image_path})
    img_t = data["img"]                                    # (C, X, Y, Z)

    # Reconstruct affine from original header
    orig      = nib.load(image_path)
    new_affine = compute_resized_affine(
        orig.affine, orig.shape, tuple(img_t.shape[1:])
    )

    # Prepare inputs
    img        = img_t.unsqueeze(0).to(device)             # (1, 1, X, Y, Z)
    img.requires_grad_(True)
    mod_tensor = torch.tensor([int(modality)], dtype=torch.long, device=device)

    # Predict class
    with torch.no_grad():
        pred_class = int(torch.argmax(model(img, mod_tensor), dim=1).item())

    target_class = pred_class if target_mode == "pred" else int(true_class)

    # Wrapper that broadcasts modality index to batch size B
    def forward_func(img_only: torch.Tensor) -> torch.Tensor:
        mod = mod_tensor.expand(img_only.size(0))
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                return model(img_only, mod)
        return model(img_only, mod)

    # SmoothGrad attribution
    attributions = NoiseTunnel(Saliency(forward_func)).attribute(
        img,
        nt_samples=nsamples,
        nt_type="smoothgrad",
        stdevs=sigma,
        target=target_class,
        nt_samples_batch_size=max(1, int(nt_batch)),
    )

    # Absolute value + normalise to [0, 1]
    attr = np.abs(attributions.detach().cpu().squeeze().numpy().astype(np.float32))
    maxv = float(attr.max()) if attr.size > 0 else 0.0
    if maxv > 0:
        attr /= (maxv + 1e-8)

    # Output filename derived from subject folder name
    stem0   = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]
    stem    = f"{name_suffix}__{stem0}" if name_suffix else stem0
    nii_out = os.path.join(save_dir, f"saliency_{stem}.nii.gz")

    # Save NIfTI with correct affine
    sal_img = nib.Nifti1Image(attr, new_affine)
    sal_img.set_qform(new_affine, code=1)
    sal_img.set_sform(new_affine, code=1)
    nib.save(sal_img, nii_out)

    # PNG overlays on MNI152 template
    if save_overlays:
        t1     = datasets.load_mni152_template()
        sal_rs = resample_to_img(
            sal_img, t1,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        save_overlay_pngs(
            t1_img=t1,
            saliency_rs_img=sal_rs,
            out_prefix=(f"{stem}_mod{modality}_true{true_class}"
                        f"_pred{pred_class}_tgt{target_class}"),
            outdir=save_dir,
        )

    del attributions, img
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return nii_out, pred_class, target_class


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute SmoothGrad saliency maps for PET images listed in a CSV. "
            "Outputs NIfTI saliency maps and optional PNG overlays on MNI152."
        )
    )
    p.add_argument("--exp_name",    type=str, required=True,
                   help="Experiment folder name (must match training)")
    p.add_argument("--model_name",  type=str, default="sfcn",
                   help="Model architecture — only 'sfcn' is supported")
    p.add_argument("--csv",         type=str, required=True,
                   help="CSV with columns: Filepath, Modality, class_label")
    p.add_argument("--save_dir",    type=str, required=True,
                   help="Output directory for saliency maps and overlays")
    p.add_argument("--max_rows",    type=int, default=None,
                   help="Process only the first N rows (for quick testing)")
    p.add_argument("--nsamples",    type=int, default=20,
                   help="Number of noisy samples for SmoothGrad (default: 20)")
    p.add_argument("--sigma",       type=float, default=0.15,
                   help="Gaussian noise std for SmoothGrad (default: 0.15)")
    p.add_argument("--device",      type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--seed",        type=int, default=1)
    p.add_argument("--nt_batch",    type=int, default=1,
                   help="Batch size for NoiseTunnel forward passes (default: 1)")
    p.add_argument("--target_mode", type=str, default="pred",
                   choices=["pred", "true"],
                   help="Attribute toward 'pred' (predicted) or 'true' class")
    p.add_argument("--resize_x",    type=int, default=None,
                   help="Target X voxels (for affine reconstruction only)")
    p.add_argument("--resize_y",    type=int, default=None,
                   help="Target Y voxels (for affine reconstruction only)")
    p.add_argument("--resize_z",    type=int, default=None,
                   help="Target Z voxels (for affine reconstruction only)")
    p.add_argument("--no_overlays", action="store_true",
                   help="Skip PNG overlay generation (save NIfTI only)")
    return p.parse_args()


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args   = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    model = build_model(args.exp_name, args.model_name, device)

    df = pd.read_csv(args.csv)
    required = {"Filepath", "Modality", "class_label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Input CSV must contain columns: {required}. "
            f"Found: {list(df.columns)}"
        )
    if args.max_rows is not None:
        df = df.iloc[:args.max_rows].copy()

    resize_xyz = (
        (args.resize_x, args.resize_y, args.resize_z)
        if args.resize_x and args.resize_y and args.resize_z
        else (cfg.params["imagex"], cfg.params["imagey"], cfg.params["imagez"])
    )

    os.makedirs(args.save_dir, exist_ok=True)
    ok, fail = 0, 0

    for i, row in df.iterrows():
        fpath       = str(row["Filepath"])
        modality    = int(row["Modality"])
        true_class  = int(row["class_label"])
        name_prefix = os.path.basename(os.path.dirname(fpath))

        print(f"[{i+1}/{len(df)}] {os.path.basename(fpath)} | "
              f"modality={modality} | true={true_class} | "
              f"target_mode={args.target_mode}")

        try:
            explain_one_case(
                model=model, device=device,
                image_path=fpath, modality=modality, true_class=true_class,
                save_dir=args.save_dir, nsamples=args.nsamples,
                sigma=args.sigma, resize_xyz=resize_xyz,
                nt_batch=args.nt_batch, name_suffix=name_prefix,
                target_mode=args.target_mode,
                save_overlays=(not args.no_overlays),
            )
            ok += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                print("WARNING: CUDA out of memory. Retrying on CPU...")
                torch.cuda.empty_cache()
                gc.collect()
                cpu_device = torch.device("cpu")
                cpu_model  = build_model(args.exp_name, args.model_name, cpu_device)
                try:
                    explain_one_case(
                        model=cpu_model, device=cpu_device,
                        image_path=fpath, modality=modality, true_class=true_class,
                        save_dir=args.save_dir, nsamples=args.nsamples,
                        sigma=args.sigma, resize_xyz=resize_xyz,
                        nt_batch=1, name_suffix=name_prefix,
                        target_mode=args.target_mode,
                        save_overlays=(not args.no_overlays),
                    )
                    ok += 1
                except Exception as e2:
                    print(f"ERROR: CPU fallback also failed: {e2}")
                    fail += 1
                finally:
                    del cpu_model
                    gc.collect()
            else:
                print(f"ERROR: {e}")
                fail += 1

        except Exception as e:
            print(f"ERROR: {e}")
            fail += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone.  Successful: {ok} | Failed: {fail} | "
          f"Output: {args.save_dir}")


if __name__ == "__main__":
    main()