# A Unified Deep Learning Framework for Alzheimer's Disease Classification Using 18F-FDG and Amyloid PET

This repository contains the full training, evaluation, and explainability pipeline used in:

> **[Paper title]** — *[Authors]*. *[Journal]*, [Year].
> DOI: [doi]

## Overview

We train and evaluate three 3-D convolutional models (SFCN backbone) for binary AD/CN classification from PET neuroimaging:

| Model | Input | Description |
|---|---|---|
| **Unified** | FDG-PET + Amyloid-PET | Single model handling both tracers via modality embedding |
| **FDG baseline** | FDG-PET only | Single-tracer unimodal baseline |
| **AMY baseline** | Amyloid-PET only | Single-tracer unimodal baseline |

Each model is trained over **25 repeated random stratified splits** (seeds 42–66) to report stable mean ± SD estimates of accuracy, AUROC, sensitivity, and specificity.

---

## Repository structure

```
.
├── generate_splits.py          # Generate 25 stratified splits for all 3 models
├── train_iteration_unified.py  # Train one iteration of the unified model
├── train_iteration_single.py   # Train one iteration of FDG or AMY baseline
├── aggregate_results.py        # Aggregate per-iteration CSVs into summary tables
├── statics.py                  # Mann-Whitney U tests: unified vs baselines (aggregate)
├── statics_1.py                # Mann-Whitney U tests: unified per-tracer vs baseline
├── saliency_maps.py            # SmoothGrad saliency maps via Captum
├── verification.py             # Pre-flight data and split integrity checks
├── SFCN_Class.py               # SFCN classifier architecture
├── SFCN_Encoder.py             # SFCN feature encoder (used by unified model)
├── utils.py                    # Model factory (get_model)
├── config_file.py              # Shared image-size parameters
├── df_both_example.csv         # Example dataset CSV (10 rows, relative paths)
└── requirements.txt
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥ 2.0, MONAI ≥ 1.3, scikit-learn, scipy, nibabel, nilearn, captum.

---

## Data availability

Images were obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**
database ([adni.loni.usc.edu](https://adni.loni.usc.edu)).
ADNI is funded by the National Institute on Aging (NIA) and the National Institute
of Biomedical Imaging and Bioengineering (NIBIB).
Researchers may apply for access at:
[https://adni.loni.usc.edu/data-samples/access-data/](https://adni.loni.usc.edu/data-samples/access-data/)

### Dataset summary

| Modality | Subjects | CN | AD |
|---|---|---|---|
| FDG-PET (Modality = 0) | 471 | 260 | 211 |
| Amyloid-PET (Modality = 1) | 245 | 130 | 115 |
| **Total** | **601** | **390** | **326** |

> Note: some subjects have both FDG and AMY scans (601 unique subjects, 716 total rows).

### Preprocessing pipeline

All PET images underwent a harmonized preprocessing pipeline to ensure voxel-wise
spatial correspondence and intensity comparability across subjects and tracers:

1. **SUVR normalization** — standardized uptake value ratio intensity normalization using a tracer-specific reference region
2. **Brain masking** — applied to all images
3. **Spatial smoothing** — applied as final step
4. **Resampling** — all volumes resampled to 128 × 128 × 128 voxels at inference time via MONAI

Tracer-specific details (frame averaging window, reference region) are described in the paper.
The preprocessing steps are reflected in the filename convention:
- FDG: `pet_mni_1mm.nii_suvr.nii_smooth.nii.gz`
- AMY: `pet_mni_1mm.nii_suvr.nii_brain.nii_smooth.nii.gz`

---

## Data format

All scripts expect a CSV file named `df_both.csv` with the following columns:

| Column | Type | Description |
|---|---|---|
| `subject_id` | str | ADNI subject identifier (e.g. `002_S_0295`) |
| `Filepath` | str | Absolute path to the preprocessed NIfTI (.nii.gz) image |
| `class_label` | int | 0 = CN (cognitively normal), 1 = AD |
| `Modality` | int | 0 = FDG-PET, 1 = Amyloid-PET |

A subject with both modalities appears as **two rows** (one per tracer).
See `df_both_example.csv` for a 10-row example with the expected structure.

Place `df_both.csv` inside the unified experiment folder before running any script:
```
experiments/
└── exp_unified/
    └── df_both.csv   ← required here; shared by all three models
```

---

## Usage

### Step 0 — Verify data integrity (optional but recommended)

```bash
python verification.py \
    --base_dir        experiments \
    --expname_unified exp_unified \
    --expname_fdg     exp_fdg \
    --expname_amy     exp_amy \
    --n_iterations    25
```

### Step 1 — Generate splits (run once on CPU)

```bash
python generate_splits.py \
    --base_dir        experiments \
    --expname_unified exp_unified \
    --expname_fdg     exp_fdg \
    --expname_amy     exp_amy \
    --seed            42 \
    --n_iterations    25
```

### Step 2 — Train (SLURM array or sequential loop)

**Unified model:**
```bash
# SLURM: --array=1-25
python train_iteration_unified.py \
    --expname   exp_unified \
    --base_dir  experiments \
    --iteration $SLURM_ARRAY_TASK_ID \
    --seed      42
```

**FDG baseline:**
```bash
python train_iteration_single.py \
    --expname          exp_fdg \
    --base_dir         experiments \
    --target_modality  fdg \
    --iteration        $SLURM_ARRAY_TASK_ID \
    --seed             42
```

**AMY baseline:**
```bash
python train_iteration_single.py \
    --expname          exp_amy \
    --base_dir         experiments \
    --target_modality  amy \
    --iteration        $SLURM_ARRAY_TASK_ID \
    --seed             42
```

### Step 3 — Aggregate results (run once after all iterations finish)

```bash
python aggregate_results.py \
    --base_dir        experiments \
    --expname_unified exp_unified \
    --expname_fdg     exp_fdg \
    --expname_amy     exp_amy \
    --n_iterations    25
```

### Step 4 — Statistical tests

```bash
# Unified vs baselines on the overall test set
python statics.py \
    --unified experiments/exp_unified/summary_unified.csv \
    --fdg     experiments/exp_fdg/summary_fdg.csv \
    --amy     experiments/exp_amy/summary_amy.csv \
    --output  mann_whitney_results.csv

# Unified per-tracer subset vs each corresponding baseline
python statics_1.py \
    --unified experiments/exp_unified/summary_unified.csv \
    --fdg     experiments/exp_fdg/summary_fdg.csv \
    --amy     experiments/exp_amy/summary_amy.csv \
    --output  mann_whitney_subset_results.csv
```

### Step 5 — Saliency maps (SmoothGrad, optional)

```bash
python saliency_maps.py \
    --exp_name   exp_unified \
    --model_name sfcn \
    --csv        /path/to/test_subjects.csv \
    --save_dir   /path/to/saliency_output \
    --nsamples   20 \
    --sigma      0.15 \
    --device     cuda
```

The `--csv` file must contain columns: `Filepath`, `Modality`, `class_label`.
Set the `EXPERIMENT_BASE_DIR` environment variable to point to your experiments root:
```bash
export EXPERIMENT_BASE_DIR=/path/to/experiments
```

---

## Statistical analysis details

All Mann-Whitney U tests are **unpaired** (two-sided). Although the same base seed is
used across models, the splits are drawn from different subject populations (unified
uses all subjects; FDG/AMY baselines use their respective subsets), so test sets are
**not identical** across models for the same seed. This invalidates a paired Wilcoxon test.

Effect size is reported as *r* = |Z| / √(N₁ + N₂), with bootstrap 95 % CIs on the
difference of medians (10 000 resamples, seed = 0).

---

## Citation

If you use this code, please cite:

```bibtex
@article{[cite_key],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[doi]}
}
```

---

## License

[MIT / Apache-2.0 / CC BY 4.0 — choose one]